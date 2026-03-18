package ai.koog.prompt.executor.clients.openai.base

import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.openai.base.models.Content
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIBaseLLMResponse
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIBaseLLMStreamResponse
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIFunction
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIMessage
import ai.koog.prompt.executor.clients.openai.base.models.OpenAITool
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIToolCall
import ai.koog.prompt.executor.clients.openai.base.models.OpenAIToolChoice
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.LLMChoice
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.streaming.StreamFrame
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.time.Clock
import kotlin.time.Instant

class ToMessageResponsesTest {

    private data class FakeResponse(
        override val id: String = "id",
        override val model: String = "model",
        override val created: Long = 0L,
    ) : OpenAIBaseLLMResponse

    private data class FakeStreamResponse(
        override val id: String = "id",
        override val model: String = "model",
        override val created: Long = 0L,
    ) : OpenAIBaseLLMStreamResponse

    private val fakeProvider = object : LLMProvider("test", "Test") {}

    /**
     * Minimal test subclass that exposes the protected [toMessageResponses] method for unit testing.
     */
    private val client = object : AbstractOpenAILLMClient<FakeResponse, FakeStreamResponse>(
        apiKey = "test",
        settings = object : OpenAIBaseSettings("http://localhost", "/test") {},
        clock = Clock.System,
        logger = KotlinLogging.logger("test"),
        toolsConverter = OpenAICompatibleToolDescriptorSchemaGenerator(),
    ) {
        fun callToMessageResponses(
            message: OpenAIMessage,
            finishReason: String?,
            metaInfo: ResponseMetaInfo,
        ) = message.toMessageResponses(finishReason, metaInfo)

        override fun llmProvider(): LLMProvider = fakeProvider

        override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
            error("not used in this test")

        override fun serializeProviderChatRequest(
            messages: List<OpenAIMessage>,
            model: LLModel,
            tools: List<OpenAITool>?,
            toolChoice: OpenAIToolChoice?,
            params: LLMParams,
            stream: Boolean,
        ): String = error("not used in this test")

        override fun processProviderChatResponse(response: FakeResponse): List<LLMChoice> =
            error("not used in this test")

        override fun decodeStreamingResponse(data: String): FakeStreamResponse =
            error("not used in this test")

        override fun decodeResponse(data: String): FakeResponse =
            error("not used in this test")

        override fun processStreamingResponse(response: Flow<FakeStreamResponse>): Flow<StreamFrame> =
            error("not used in this test")
    }

    private val metaInfo = ResponseMetaInfo(timestamp = Instant.DISTANT_PAST)

    @Test
    fun testToolCallsOnlyProducesToolCallMessages() {
        val message = OpenAIMessage.Assistant(
            toolCalls = listOf(
                OpenAIToolCall(id = "call_1", function = OpenAIFunction(name = "my_tool", arguments = """{"key":"value"}"""))
            )
        )

        val result = client.callToMessageResponses(message, finishReason = null, metaInfo = metaInfo)

        assertEquals(1, result.size)
        assertIs<Message.Tool.Call>(result[0])
        assertEquals("call_1", (result[0] as Message.Tool.Call).id)
        assertEquals("my_tool", (result[0] as Message.Tool.Call).tool)
    }

    /**
     * Reproduces the bug reported in https://github.com/DeniDoman/koog/issues/2.
     *
     * DeepSeek (and potentially other OpenAI-compatible providers) may return a choice that
     * contains BOTH a text [content] field AND [toolCalls]. Before the fix, the text content
     * was silently discarded because the toolCalls branch was handled as an exclusive condition
     * and returned early without appending the content message.
     */
    @Test
    fun testContentAndToolCallsBothPreserved() {
        val message = OpenAIMessage.Assistant(
            content = Content.Text("Let me search for that."),
            toolCalls = listOf(
                OpenAIToolCall(id = "call_1", function = OpenAIFunction(name = "search", arguments = """{"query":"test"}"""))
            )
        )

        val result = client.callToMessageResponses(message, finishReason = "tool_calls", metaInfo = metaInfo)

        assertEquals(2, result.size)
        assertIs<Message.Assistant>(result[0])
        assertEquals("Let me search for that.", (result[0] as Message.Assistant).content)
        assertIs<Message.Tool.Call>(result[1])
        assertEquals("call_1", (result[1] as Message.Tool.Call).id)
        assertEquals("search", (result[1] as Message.Tool.Call).tool)
        assertEquals("""{"query":"test"}""", (result[1] as Message.Tool.Call).content)
    }

    @Test
    fun testReasoningContentAndToolCallsPreserved() {
        val message = OpenAIMessage.Assistant(
            reasoningContent = "I will use the search tool.",
            toolCalls = listOf(
                OpenAIToolCall(id = "call_1", function = OpenAIFunction(name = "search", arguments = """{"query":"test"}"""))
            )
        )

        val result = client.callToMessageResponses(message, finishReason = "tool_calls", metaInfo = metaInfo)

        assertEquals(2, result.size)
        assertIs<Message.Reasoning>(result[0])
        assertEquals("I will use the search tool.", (result[0] as Message.Reasoning).content)
        assertIs<Message.Tool.Call>(result[1])
        assertEquals("call_1", (result[1] as Message.Tool.Call).id)
    }

    @Test
    fun testReasoningAndTextContentAndToolCallsAllPreserved() {
        val message = OpenAIMessage.Assistant(
            reasoningContent = "Thinking about what tool to use.",
            content = Content.Text("Calling the search tool now."),
            toolCalls = listOf(
                OpenAIToolCall(id = "call_1", function = OpenAIFunction(name = "search", arguments = """{"query":"test"}"""))
            )
        )

        val result = client.callToMessageResponses(message, finishReason = "tool_calls", metaInfo = metaInfo)

        assertEquals(3, result.size)
        assertIs<Message.Reasoning>(result[0])
        assertEquals("Thinking about what tool to use.", (result[0] as Message.Reasoning).content)
        assertIs<Message.Assistant>(result[1])
        assertEquals("Calling the search tool now.", (result[1] as Message.Assistant).content)
        assertIs<Message.Tool.Call>(result[2])
        assertEquals("call_1", (result[2] as Message.Tool.Call).id)
    }
}
