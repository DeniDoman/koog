package ai.koog.agents.memory

import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.context.AIAgentLLMContext
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.memory.config.MemoryScopeType
import ai.koog.agents.memory.config.MemoryScopesProfile
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.memory.model.MultipleFacts
import ai.koog.agents.memory.model.SingleFact
import ai.koog.agents.memory.model.TokenBudget
import ai.koog.agents.memory.model.estimateTokens
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.testing.tools.MockEnvironment
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import io.mockk.coEvery
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.Serializable
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Clock
import kotlin.time.Instant

@OptIn(InternalAgentsApi::class)
class TokenBudgetTest {

    @Serializable
    data object UserSubject : MemorySubject() {
        override val name: String = "user"
        override val promptDescription: String = "User preferences"
        override val priorityLevel: Int = 2
    }

    private val testModel = mockk<LLModel> {
        every { id } returns "test-model"
        every { provider } returns mockk<LLMProvider>()
    }

    private val testClock: Clock = object : Clock {
        override fun now(): Instant = Instant.parse("2023-01-01T00:00:00Z")
    }

    // -- TokenBudget unit tests --

    @Test
    fun testTokenBudgetValidation() {
        assertFailsWith<IllegalArgumentException> { TokenBudget(maxFacts = 0) }
        assertFailsWith<IllegalArgumentException> { TokenBudget(maxFacts = -1) }
        assertFailsWith<IllegalArgumentException> { TokenBudget(maxTokens = 0) }
        assertFailsWith<IllegalArgumentException> { TokenBudget(maxTokens = -5) }

        // Valid budgets should not throw
        TokenBudget(maxFacts = 1, maxTokens = 1)
        TokenBudget.Unlimited
    }

    @Test
    fun testEstimateTokens() {
        assertEquals(0, "".estimateTokens())
        assertEquals(1, "abc".estimateTokens()) // 3 chars / 4 = 0.75 rounds up to 1
        assertEquals(1, "abcd".estimateTokens()) // 4 chars / 4 = 1
        assertEquals(3, "Hello world!".estimateTokens()) // 12 chars / 4 = 3
    }

    // -- applyTokenBudget unit tests --

    @Test
    fun testApplyTokenBudgetUnlimitedReturnsAllFacts() {
        val concept = Concept("test", "desc", FactType.SINGLE)
        val facts = (1..100).map {
            SingleFact(concept = concept, value = "Fact number $it with some text", timestamp = it.toLong())
        }

        val memory = AgentMemory(
            agentMemory = mockk(),
            scopesProfile = MemoryScopesProfile()
        )

        val result = memory.applyTokenBudget(facts, TokenBudget.Unlimited)
        assertEquals(100, result.size, "Unlimited budget should return all facts")
    }

    @Test
    fun testApplyTokenBudgetMaxFactsCapsCount() {
        val concept = Concept("test", "desc", FactType.SINGLE)
        val facts = (1..20).map {
            SingleFact(concept = concept, value = "Fact $it", timestamp = it.toLong())
        }

        val memory = AgentMemory(
            agentMemory = mockk(),
            scopesProfile = MemoryScopesProfile()
        )

        val result = memory.applyTokenBudget(facts, TokenBudget(maxFacts = 5))
        assertEquals(5, result.size, "Should cap at maxFacts=5")
        // Should prefer newest facts (highest timestamps)
        assertTrue(result.all { (it as SingleFact).timestamp >= 16 },
            "Should keep the 5 most recent facts (timestamps 16-20)")
    }

    @Test
    fun testApplyTokenBudgetMaxTokensCapsContent() {
        val concept = Concept("k", "desc", FactType.SINGLE)
        // Each fact: keyword "k" (1 token) + value "x".repeat(40) (10 tokens) = ~11 tokens
        val facts = (1..10).map {
            SingleFact(concept = concept, value = "x".repeat(40), timestamp = it.toLong())
        }

        val memory = AgentMemory(
            agentMemory = mockk(),
            scopesProfile = MemoryScopesProfile()
        )

        // Budget for ~25 tokens should fit about 2 facts (each ~11 tokens)
        val result = memory.applyTokenBudget(facts, TokenBudget(maxTokens = 25))
        assertTrue(result.size in 1..3, "Should include ~2 facts within 25 token budget, got ${result.size}")
    }

    @Test
    fun testApplyTokenBudgetPrefersNewerFacts() {
        val concept = Concept("test", "desc", FactType.SINGLE)
        val facts = listOf(
            SingleFact(concept = concept, value = "old fact", timestamp = 100L),
            SingleFact(concept = concept, value = "newer fact", timestamp = 200L),
            SingleFact(concept = concept, value = "newest fact", timestamp = 300L),
        )

        val memory = AgentMemory(
            agentMemory = mockk(),
            scopesProfile = MemoryScopesProfile()
        )

        val result = memory.applyTokenBudget(facts, TokenBudget(maxFacts = 2))
        assertEquals(2, result.size)
        assertEquals(300L, result[0].timestamp, "First should be newest")
        assertEquals(200L, result[1].timestamp, "Second should be second-newest")
    }

    @Test
    fun testApplyTokenBudgetTrimsMultipleFactsValues() {
        val concept = Concept("items", "desc", FactType.MULTIPLE)
        val fact = MultipleFacts(
            concept = concept,
            values = (1..50).map { "Item number $it with a moderately long description text" },
            timestamp = 1L
        )

        val memory = AgentMemory(
            agentMemory = mockk(),
            scopesProfile = MemoryScopesProfile()
        )

        // Very tight token budget should trim values within the MultipleFacts
        val result = memory.applyTokenBudget(listOf(fact), TokenBudget(maxTokens = 100))
        assertTrue(result.isNotEmpty(), "Should include at least the fact")
        val trimmed = result.first() as MultipleFacts
        assertTrue(trimmed.values.size < 50,
            "Should trim values within MultipleFacts, got ${trimmed.values.size}")
    }

    // -- Integration test: reproduces the bug scenario from the issue --

    @Test
    fun testUnboundedFactGrowthWithoutBudget() = runTest {
        // Simulate the CustomerSupportAgent scenario: 12 sessions, ~4 MultipleFacts each
        val sessionsCount = 12
        val conceptsPerSession = listOf(
            Concept("user-preferences", "User communication preferences", FactType.MULTIPLE),
            Concept("user-issues", "Issue descriptions and resolutions", FactType.MULTIPLE),
            Concept("diagnostic-results", "Device diagnostics and error codes", FactType.MULTIPLE),
            Concept("organization-solutions", "Organization-wide solutions", FactType.MULTIPLE),
        )

        val allFacts = mutableListOf<MultipleFacts>()
        for (session in 1..sessionsCount) {
            for (concept in conceptsPerSession) {
                allFacts.add(
                    MultipleFacts(
                        concept = concept,
                        values = listOf(
                            "Session $session detail 1 for ${concept.keyword}: some moderately long text describing the fact",
                            "Session $session detail 2 for ${concept.keyword}: another piece of relevant information here",
                            "Session $session detail 3 for ${concept.keyword}: yet more context that the agent remembered",
                        ),
                        timestamp = (session * 1000).toLong()
                    )
                )
            }
        }

        // Total: 12 sessions * 4 concepts * 3 values = 144 individual fact values
        assertEquals(48, allFacts.size, "Should have 48 MultipleFacts entries (12 sessions * 4 concepts)")

        val memoryProvider = mockk<AgentMemoryProvider>()
        coEvery { memoryProvider.loadAll(any(), any()) } returns allFacts

        val promptExecutor = mockk<PromptExecutor>()
        val capturedPrompts = mutableListOf<Prompt>()
        coEvery { promptExecutor.execute(capture(capturedPrompts), any()) } returns listOf(
            mockk<Message.Response> { every { content } returns "OK" }
        )

        val llm = AIAgentLLMContext(
            tools = emptyList(),
            prompt = prompt("test") {},
            model = testModel,
            responseProcessor = null,
            promptExecutor = promptExecutor,
            environment = MockEnvironment(ToolRegistry.EMPTY, promptExecutor, ai.koog.serialization.kotlinx.KotlinxSerializer()),
            config = AIAgentConfig(Prompt.Empty, testModel, 100),
            clock = testClock
        )

        val memory = AgentMemory(
            agentMemory = memoryProvider,
            scopesProfile = MemoryScopesProfile(MemoryScopeType.AGENT to "test-agent")
        )

        // WITHOUT budget: all 48 fact entries are loaded, producing a very large prompt
        memory.loadAllFactsToAgent(
            llm = llm,
            scopes = listOf(MemoryScopeType.AGENT),
            subjects = listOf(UserSubject),
        )

        // Measure the injected content by reading the prompt messages
        val promptMessages = llm.readSession { prompt.messages }
        val totalContent = promptMessages.joinToString("\n") { it.content }
        val totalChars = totalContent.length

        // Issue states ~8000 tokens (>32000 chars) for 12 sessions — verify it's large
        assertTrue(totalChars > 5000,
            "Without budget, injected content should be large (was $totalChars chars)")
    }

    @Test
    fun testTokenBudgetCapsFactsInLoadAllFactsToAgent() = runTest {
        // Same setup as above but WITH a token budget
        val sessionsCount = 12
        val conceptsPerSession = listOf(
            Concept("user-preferences", "User communication preferences", FactType.MULTIPLE),
            Concept("user-issues", "Issue descriptions and resolutions", FactType.MULTIPLE),
            Concept("diagnostic-results", "Device diagnostics and error codes", FactType.MULTIPLE),
            Concept("organization-solutions", "Organization-wide solutions", FactType.MULTIPLE),
        )

        val allFacts = mutableListOf<MultipleFacts>()
        for (session in 1..sessionsCount) {
            for (concept in conceptsPerSession) {
                allFacts.add(
                    MultipleFacts(
                        concept = concept,
                        values = listOf(
                            "Session $session detail 1 for ${concept.keyword}: some moderately long text describing the fact",
                            "Session $session detail 2 for ${concept.keyword}: another piece of relevant information here",
                            "Session $session detail 3 for ${concept.keyword}: yet more context that the agent remembered",
                        ),
                        timestamp = (session * 1000).toLong()
                    )
                )
            }
        }

        val memoryProvider = mockk<AgentMemoryProvider>()
        coEvery { memoryProvider.loadAll(any(), any()) } returns allFacts

        val promptExecutor = mockk<PromptExecutor>()
        coEvery { promptExecutor.execute(any(), any()) } returns listOf(
            mockk<Message.Response> { every { content } returns "OK" }
        )

        val llm = AIAgentLLMContext(
            tools = emptyList(),
            prompt = prompt("test") {},
            model = testModel,
            responseProcessor = null,
            promptExecutor = promptExecutor,
            environment = MockEnvironment(ToolRegistry.EMPTY, promptExecutor, ai.koog.serialization.kotlinx.KotlinxSerializer()),
            config = AIAgentConfig(Prompt.Empty, testModel, 100),
            clock = testClock
        )

        val memory = AgentMemory(
            agentMemory = memoryProvider,
            scopesProfile = MemoryScopesProfile(MemoryScopeType.AGENT to "test-agent")
        )

        // WITH budget: cap at 8 facts and ~2000 tokens
        memory.loadAllFactsToAgent(
            llm = llm,
            scopes = listOf(MemoryScopeType.AGENT),
            subjects = listOf(UserSubject),
            tokenBudget = TokenBudget(maxFacts = 8, maxTokens = 2000)
        )

        val promptMessages = llm.readSession { prompt.messages }
        val totalContent = promptMessages.joinToString("\n") { it.content }
        val totalChars = totalContent.length
        val estimatedTokens = totalContent.estimateTokens()

        // Verify budget is respected
        assertTrue(estimatedTokens <= 2500,
            "With token budget of 2000, estimated tokens should be capped (was $estimatedTokens)")
        assertTrue(totalChars < 10000,
            "With budget, injected content should be much smaller (was $totalChars chars)")
    }

    @Test
    fun testTokenBudgetCapsFactsInLoadFactsToAgent() = runTest {
        val concept = Concept("user-issues", "Issue descriptions", FactType.MULTIPLE)
        val facts = (1..20).map { session ->
            MultipleFacts(
                concept = concept,
                values = listOf("Issue from session $session: detailed description of the problem"),
                timestamp = (session * 1000).toLong()
            )
        }

        val memoryProvider = mockk<AgentMemoryProvider>()
        coEvery { memoryProvider.load(concept, any(), any()) } returns facts

        val promptExecutor = mockk<PromptExecutor>()
        coEvery { promptExecutor.execute(any(), any()) } returns listOf(
            mockk<Message.Response> { every { content } returns "OK" }
        )

        val llm = AIAgentLLMContext(
            tools = emptyList(),
            prompt = prompt("test") {},
            model = testModel,
            responseProcessor = null,
            promptExecutor = promptExecutor,
            environment = MockEnvironment(ToolRegistry.EMPTY, promptExecutor, ai.koog.serialization.kotlinx.KotlinxSerializer()),
            config = AIAgentConfig(Prompt.Empty, testModel, 100),
            clock = testClock
        )

        val memory = AgentMemory(
            agentMemory = memoryProvider,
            scopesProfile = MemoryScopesProfile(MemoryScopeType.AGENT to "test-agent")
        )

        // Load with budget of max 5 facts
        memory.loadFactsToAgent(
            llm = llm,
            concept = concept,
            scopes = listOf(MemoryScopeType.AGENT),
            subjects = listOf(UserSubject),
            tokenBudget = TokenBudget(maxFacts = 5)
        )

        val promptMessages = llm.readSession { prompt.messages }
        val totalContent = promptMessages.joinToString("\n") { it.content }

        // With 20 facts capped at 5, only the 5 newest sessions (16-20) should appear
        for (session in 16..20) {
            assertTrue(totalContent.contains("session $session:"),
                "Should contain newest session $session")
        }
        // Use "session N:" pattern to avoid matching "session 1" inside "session 10"
        for (session in 1..9) {
            assertTrue(!totalContent.contains("session $session:"),
                "Should NOT contain old session $session")
        }
    }
}
