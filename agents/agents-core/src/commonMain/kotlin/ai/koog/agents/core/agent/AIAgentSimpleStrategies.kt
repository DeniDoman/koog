package ai.koog.agents.core.agent

import ai.koog.agents.core.agent.entity.AIAgentGraphStrategy
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.node
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeExecuteMultipleTools
import ai.koog.agents.core.dsl.extension.nodeExecuteTool
import ai.koog.agents.core.dsl.extension.nodeLLMRequest
import ai.koog.agents.core.dsl.extension.nodeLLMRequestMultiple
import ai.koog.agents.core.dsl.extension.nodeLLMSendMultipleToolResults
import ai.koog.agents.core.dsl.extension.nodeLLMSendToolResult
import ai.koog.agents.core.dsl.extension.onAssistantMessage
import ai.koog.agents.core.dsl.extension.onMultipleAssistantMessages
import ai.koog.agents.core.dsl.extension.onMultipleToolCalls
import ai.koog.agents.core.dsl.extension.onToolCall
import ai.koog.agents.core.environment.ReceivedToolResult
import ai.koog.agents.core.environment.result
import ai.koog.prompt.message.Message
import ai.koog.prompt.streaming.toMessageResponses
import kotlinx.coroutines.flow.toList
import kotlin.jvm.JvmOverloads

/**
 * Creates a single-run strategy for an AI agent.
 * This strategy defines a simple execution flow where the agent processes input,
 * calls tools, and sends results back to the agent.
 * The flow consists of the following steps:
 * 1. Start the agent.
 * 2. Call the LLM with the input.
 * 3. Execute a tool based on the LLM's response.
 * 4. Send the tool result back to the LLM.
 * 5. Repeat until LLM indicates no further tool calls are needed or the agent finishes.
 * @param runMode The mode in which the single-run strategy should operate. Defaults to [ToolCalls.SEQUENTIAL].
 *                - [ToolCalls.SEQUENTIAL]: Executes multiple tool calls sequentially.
 *                - [ToolCalls.PARALLEL]: Executes multiple tool calls in parallel.
 *                - [ToolCalls.SINGLE_RUN_SEQUENTIAL]: Executes a single tool call per step.
 * @param stream When `true`, the LLM is queried via a streaming request (`requestLLMStreaming`) instead of a regular
 *               one. This makes the agent emit streaming pipeline events (for example,
 *               `onLLMStreamingFrameReceived`) so installed event handlers can react to text deltas in real time.
 *               The streamed frames are collected into [Message.Response]s, so tool calling keeps working exactly as
 *               in the non-streaming flow. Defaults to `false`.
 *               Note: streaming always uses the multiple-tool-call flow, so [ToolCalls.SINGLE_RUN_SEQUENTIAL] behaves
 *               like [ToolCalls.SEQUENTIAL] when `stream` is `true`.
 * @return An instance of AIAgentStrategy configured according to the specified run mode.
 */
@JvmOverloads
public fun singleRunStrategy(
    runMode: ToolCalls = ToolCalls.SEQUENTIAL,
    stream: Boolean = false,
): AIAgentGraphStrategy<String, String> =
    when (runMode) {
        ToolCalls.SEQUENTIAL -> if (stream) singleRunStreaming(parallelTools = false) else singleRunWithParallelAbility(false)
        ToolCalls.PARALLEL -> if (stream) singleRunStreaming(parallelTools = true) else singleRunWithParallelAbility(true)
        ToolCalls.SINGLE_RUN_SEQUENTIAL -> if (stream) singleRunStreaming(parallelTools = false) else singleRunModeStrategy()
    }

private fun singleRunWithParallelAbility(parallelTools: Boolean) = strategy("single_run_sequential") {
    val nodeCallLLM by nodeLLMRequestMultiple()
    val nodeExecuteTool by nodeExecuteMultipleTools(parallelTools = parallelTools)
    val nodeSendToolResult by nodeLLMSendMultipleToolResults()

    edge(nodeStart forwardTo nodeCallLLM)
    edge(nodeCallLLM forwardTo nodeExecuteTool onMultipleToolCalls { true })
    edge(
        nodeCallLLM forwardTo nodeFinish
            onMultipleAssistantMessages { true }
            transformed { it.joinToString("\n") { message -> message.content } }
    )

    edge(nodeExecuteTool forwardTo nodeSendToolResult)

    edge(nodeSendToolResult forwardTo nodeExecuteTool onMultipleToolCalls { true })

    edge(
        nodeSendToolResult forwardTo nodeFinish
            onMultipleAssistantMessages { true }
            transformed { it.joinToString("\n") { message -> message.content } }
    )
}

private fun singleRunModeStrategy() = strategy("single_run") {
    val nodeCallLLM by nodeLLMRequest()
    val nodeExecuteTool by nodeExecuteTool()
    val nodeSendToolResult by nodeLLMSendToolResult()

    edge(nodeStart forwardTo nodeCallLLM)
    edge(nodeCallLLM forwardTo nodeExecuteTool onToolCall { true })
    edge(nodeCallLLM forwardTo nodeFinish onAssistantMessage { true })
    edge(nodeExecuteTool forwardTo nodeSendToolResult)
    edge(nodeSendToolResult forwardTo nodeFinish onAssistantMessage { true })
    edge(nodeSendToolResult forwardTo nodeExecuteTool onToolCall { true })
}

/**
 * Streaming counterpart of [singleRunWithParallelAbility].
 *
 * It has the same graph shape, but queries the LLM through [requestLLMStreaming][ai.koog.agents.core.agent.session.AIAgentLLMWriteSession.requestLLMStreaming]
 * so that streaming pipeline events are emitted while the response is produced. The streamed frames are collected and
 * converted back into [Message.Response]s via [toMessageResponses], which keeps the downstream tool-call/finish
 * branching identical to the non-streaming flow.
 */
private fun singleRunStreaming(parallelTools: Boolean) = strategy("single_run_streaming") {
    val nodeCallLLM by node<String, List<Message.Response>>("nodeCallLLMStreaming") { message ->
        llm.writeSession {
            appendPrompt {
                user(message)
            }

            requestLLMStreaming()
                .toList()
                .toMessageResponses()
                .also { appendPrompt { messages(it) } }
        }
    }
    val nodeExecuteTool by nodeExecuteMultipleTools(parallelTools = parallelTools)
    val nodeSendToolResult by node<List<ReceivedToolResult>, List<Message.Response>>("nodeSendToolResultStreaming") { results ->
        llm.writeSession {
            appendPrompt {
                tool {
                    results.forEach { result(it) }
                }
            }

            requestLLMStreaming()
                .toList()
                .toMessageResponses()
                .also { appendPrompt { messages(it) } }
        }
    }

    edge(nodeStart forwardTo nodeCallLLM)
    edge(nodeCallLLM forwardTo nodeExecuteTool onMultipleToolCalls { true })
    edge(
        nodeCallLLM forwardTo nodeFinish
            onMultipleAssistantMessages { true }
            transformed { it.joinToString("\n") { message -> message.content } }
    )

    edge(nodeExecuteTool forwardTo nodeSendToolResult)

    edge(nodeSendToolResult forwardTo nodeExecuteTool onMultipleToolCalls { true })

    edge(
        nodeSendToolResult forwardTo nodeFinish
            onMultipleAssistantMessages { true }
            transformed { it.joinToString("\n") { message -> message.content } }
    )
}

/**
 * Enum representing the modes in which a single-run strategy for an AI agent can be executed.
 *
 * These modes define how tasks or operations are processed during the agent's run:
 * - SEQUENTIAL: Multiple tool calls allowed but will be executed sequentially.
 * - PARALLEL: Tool calls executed in parallel.
 * - SINGLE: Multiple tool calls are not allowed.
 */
public enum class ToolCalls {
    SEQUENTIAL,
    PARALLEL,
    SINGLE_RUN_SEQUENTIAL
}
