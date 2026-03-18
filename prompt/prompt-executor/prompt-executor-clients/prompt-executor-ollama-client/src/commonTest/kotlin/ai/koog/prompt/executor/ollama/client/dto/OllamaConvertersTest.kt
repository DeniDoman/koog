package ai.koog.prompt.executor.ollama.client.dto

import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.ollama.client.OllamaModels
import ai.koog.prompt.message.AttachmentContent
import ai.koog.prompt.message.ContentPart
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class OllamaConvertersTest {

    private val model = OllamaModels.Meta.LLAMA_3_2

    @Test
    fun testFileAttachmentWithPlainTextContent() {
        val prompt = prompt("test") {
            user {
                text("Here is a file:")
                file(
                    ContentPart.File(
                        content = AttachmentContent.PlainText("file content here"),
                        format = "txt",
                        mimeType = "text/plain",
                        fileName = "test.txt"
                    )
                )
            }
        }

        val messages = prompt.toOllamaChatMessages(model)
        val userMessage = messages.first { it.role == "user" }

        assertTrue(userMessage.content.contains("Here is a file:"))
        assertTrue(userMessage.content.contains("file content here"))
        assertNull(userMessage.images)
    }

    @Test
    fun testFileAttachmentWithBinaryContent() {
        val binaryData = "binary file data".encodeToByteArray()
        val prompt = prompt("test") {
            user {
                text("Here is a binary file:")
                file(
                    ContentPart.File(
                        content = AttachmentContent.Binary.Bytes(binaryData),
                        format = "bin",
                        mimeType = "application/octet-stream",
                        fileName = "test.bin"
                    )
                )
            }
        }

        val messages = prompt.toOllamaChatMessages(model)
        val userMessage = messages.first { it.role == "user" }

        assertTrue(userMessage.content.contains("Here is a binary file:"))
        // Binary content should be base64 encoded
        assertTrue(userMessage.content.length > "Here is a binary file:".length)
        assertNull(userMessage.images)
    }

    @Test
    fun testFileAttachmentWithUrlContentThrowsDescriptiveError() {
        val prompt = prompt("test") {
            user {
                text("Here is a file:")
                file(
                    ContentPart.File(
                        content = AttachmentContent.URL("https://example.com/file.pdf"),
                        format = "pdf",
                        mimeType = "application/pdf",
                        fileName = "file.pdf"
                    )
                )
            }
        }

        val exception = assertFailsWith<IllegalArgumentException> {
            prompt.toOllamaChatMessages(model)
        }

        assertTrue(exception.message!!.contains("Ollama does not support URL-based file attachments"))
        assertTrue(exception.message!!.contains("textFile()"))
    }

    @Test
    fun testTextOnlyMessageConverts() {
        val prompt = prompt("test") {
            user("Hello, world!")
        }

        val messages = prompt.toOllamaChatMessages(model)
        assertEquals(1, messages.size)

        val userMessage = messages.first()
        assertEquals("user", userMessage.role)
        assertEquals("Hello, world!", userMessage.content)
        assertNull(userMessage.images)
    }
}
