#if canImport(MediaPipeTasksGenAI)
import Foundation
import MediaPipeTasksGenAI
import MediaPipeTasksGenAIC

struct InferenceModel {
    private(set) var inference: LlmInference

    init(modelPath: String,
         maxTokens: Int,
         supportedLoraRanks: [Int]? = nil,
         maxNumImages: Int = 0,
         preferredBackend: PreferredBackend? = nil,
         supportAudio: Bool = false) throws {

        let llmOptions = LlmInference.Options(modelPath: modelPath)
        llmOptions.maxTokens = maxTokens
        llmOptions.waitForWeightUploads = true

        if let supportedLoraRanks = supportedLoraRanks {
            llmOptions.supportedLoraRanks = supportedLoraRanks
        }

        if maxNumImages > 0 {
            llmOptions.maxImages = maxNumImages
        }

        if let backend = preferredBackend {
            switch backend {
            case .gpu:
                llmOptions.preferredBackend = .gpu
            case .cpu:
                llmOptions.preferredBackend = .cpu
            case .npu:
                break
            }
        }

        if supportAudio {
            llmOptions.enableAudioModality = true
        }

        self.inference = try LlmInference(options: llmOptions)
    }

    var metrics: LlmInference.Metrics {
        return inference.metrics
    }
}

final class InferenceSession {
    private let session: LlmInference.Session

    init(inference: LlmInference,
         temperature: Float,
         randomSeed: Int,
         topk: Int,
         topP: Double? = nil,
         loraPath: String? = nil,
         enableVisionModality: Bool = false,
         enableAudioModality: Bool = false) throws {

        let options = LlmInference.Session.Options()
        options.temperature = temperature
        options.randomSeed = randomSeed
        options.topk = topk

        if let topP = topP {
            options.topp = Float(topP)
        }

        if let loraPath = loraPath {
            options.loraPath = loraPath
        }

        options.enableVisionModality = enableVisionModality
        options.enableAudioModality = enableAudioModality

        do {
            let newSession = try LlmInference.Session(llmInference: inference, options: options)
            _ = try newSession.sizeInTokens(text: " ")
            self.session = newSession
        } catch {
            let fallbackOptions = LlmInference.Session.Options()
            fallbackOptions.temperature = temperature
            fallbackOptions.randomSeed = randomSeed
            fallbackOptions.topk = topk
            fallbackOptions.enableVisionModality = enableVisionModality
            fallbackOptions.enableAudioModality = enableAudioModality

            if let topP = topP {
                fallbackOptions.topp = Float(topP)
            }
            if let loraPath = loraPath {
                fallbackOptions.loraPath = loraPath
            }
            self.session = try LlmInference.Session(llmInference: inference, options: fallbackOptions)
        }
    }

    func sizeInTokens(prompt: String) throws -> Int {
        return try session.sizeInTokens(text: prompt)
    }

    func addQueryChunk(prompt: String) throws {
        try session.addQueryChunk(inputText: prompt)
    }

    func addImage(image: CGImage) throws {
        try session.addImage(image: image)
    }

    func addAudio(audio: Data) throws {
        try session.addAudio(audio: audio)
    }

    func cancelGeneration() throws {
        try session.cancelGenerateResponseAsync()
    }

    func clone() throws -> InferenceSession {
        let clonedSession = try session.clone()
        return InferenceSession(wrapping: clonedSession)
    }

    private init(wrapping session: LlmInference.Session) {
        self.session = session
    }

    func generateResponse(prompt: String? = nil) throws -> String {
        if let prompt = prompt {
            try session.addQueryChunk(inputText: prompt)
        }
        let response = try session.generateResponse()
        return response
    }

    @available(iOS 13.0.0, *)
    func generateResponseAsync(prompt: String? = nil) throws -> AsyncThrowingStream<String, any Error> {
        if let prompt = prompt {
            try session.addQueryChunk(inputText: prompt)
        }

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var fullResponse = ""
                    for try await partialResult in session.generateResponseAsync() {
                        fullResponse += partialResult
                        continuation.yield(partialResult)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    var metrics: LlmInference.Session.Metrics {
        return session.metrics
    }
}
#endif
