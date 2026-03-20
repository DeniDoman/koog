package ai.koog.agents.memory.model

/**
 * Controls how many facts (and approximate tokens) are injected into the LLM
 * prompt when loading from memory.
 *
 * Both limits are optional and applied independently — whichever is hit first wins.
 * When neither is set, the budget is effectively unlimited (backwards-compatible default).
 *
 * Token estimation uses a simple heuristic of ~4 characters per token, which is
 * a reasonable approximation for English text across most tokenizers.
 *
 * Example usage:
 * ```kotlin
 * // Cap at 20 fact entries and ~4000 tokens
 * val budget = TokenBudget(maxFacts = 20, maxTokens = 4000)
 *
 * memory.loadAllFactsToAgent(
 *     llm = llm,
 *     tokenBudget = budget
 * )
 * ```
 *
 * @property maxFacts Maximum number of individual fact entries to inject. `null` means unlimited.
 * @property maxTokens Approximate upper bound on injected tokens. `null` means unlimited.
 */
public data class TokenBudget(
    val maxFacts: Int? = null,
    val maxTokens: Int? = null,
) {
    init {
        require(maxFacts == null || maxFacts > 0) { "maxFacts must be positive, got $maxFacts" }
        require(maxTokens == null || maxTokens > 0) { "maxTokens must be positive, got $maxTokens" }
    }

    public companion object {
        /**
         * No limits — all loaded facts are injected. This is the default, preserving
         * existing behaviour for callers that don't opt in to budgeting.
         */
        public val Unlimited: TokenBudget = TokenBudget()

        /** Rough characters-per-token ratio used for estimation. */
        internal const val CHARS_PER_TOKEN: Int = 4
    }
}

/**
 * Estimates the token count of a string using a simple character-based heuristic.
 */
internal fun String.estimateTokens(): Int = (length + TokenBudget.CHARS_PER_TOKEN - 1) / TokenBudget.CHARS_PER_TOKEN
