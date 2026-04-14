import Flutter
import UIKit

@available(iOS 13.0, *)
public class FlutterGemmaPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
      let platformService = PlatformServiceImpl()
      PlatformServiceSetup.setUp(binaryMessenger: registrar.messenger(), api: platformService)

      let eventChannel = FlutterEventChannel(
        name: "flutter_gemma_stream", binaryMessenger: registrar.messenger())
      eventChannel.setStreamHandler(platformService)

      // Bundled resources method channel
      let bundledChannel = FlutterMethodChannel(
        name: "flutter_gemma_bundled",
        binaryMessenger: registrar.messenger())
      bundledChannel.setMethodCallHandler { (call, result) in
        if call.method == "getBundledResourcePath" {
          guard let args = call.arguments as? [String: Any],
                let resourceName = args["resourceName"] as? String else {
            result(FlutterError(code: "INVALID_ARGS",
                               message: "resourceName is required",
                               details: nil))
            return
          }

          // Split resourceName into name and extension
          let components = resourceName.split(separator: ".")
          let name = String(components[0])
          let ext = components.count > 1 ? String(components[1]) : ""

          // Get path from Bundle.main
          if let path = Bundle.main.path(forResource: name, ofType: ext) {
            result(path)
          } else {
            result(FlutterError(code: "NOT_FOUND",
                               message: "Resource not found in bundle: \(resourceName)",
                               details: nil))
          }
        } else {
          result(FlutterMethodNotImplemented)
        }
      }
  }
}

class PlatformServiceImpl : NSObject, PlatformService, FlutterStreamHandler {
    private var eventSink: FlutterEventSink?

    // When MediaPipe is available (Android builds via Gradle, or future Xcode fix),
    // the full inference implementation is used. When not available (current Xcode 26
    // linker bug), all methods return "not available" errors so the app still builds.
    #if canImport(MediaPipeTasksGenAI)
    private var model: InferenceModel?
    private var session: InferenceSession?
    private var _hasMediaPipe: Bool { true }
    #else
    private var _hasMediaPipe: Bool { false }
    #endif

    // Embedding model (like Android EmbeddingModel — no wrapper)
    // Device-only: TFLite has no simulator support
    #if canImport(TensorFlowLite)
    private var embeddingModel: EmbeddingModel?
    #endif

    private func _notAvailableError() -> Error {
        PigeonError(code: "MediaPipeNotAvailable", message: "AI inference not available on this platform (Xcode 26 linker limitation)", details: nil)
    }

    func createModel(
        maxTokens: Int64,
        modelPath: String,
        loraRanks: [Int64]?,
        preferredBackend: PreferredBackend?,
        maxNumImages: Int64?,
        supportAudio: Bool?,
        completion: @escaping (Result<Void, any Error>) -> Void
    ) {
        #if canImport(MediaPipeTasksGenAI)
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                self.model = try InferenceModel(
                    modelPath: modelPath,
                    maxTokens: Int(maxTokens),
                    supportedLoraRanks: loraRanks?.map(Int.init),
                    maxNumImages: Int(maxNumImages ?? 0),
                    preferredBackend: preferredBackend,
                    supportAudio: supportAudio ?? false
                )
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func closeModel(completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)
        model = nil
        #endif
        completion(.success(()))
    }

    func createSession(
        temperature: Double,
        randomSeed: Int64,
        topK: Int64,
        topP: Double?,
        loraPath: String?,
        enableVisionModality: Bool?,
        enableAudioModality: Bool?,
        systemInstruction: String?,
        enableThinking: Bool?,
        completion: @escaping (Result<Void, any Error>) -> Void
    ) {
        #if canImport(MediaPipeTasksGenAI)

        guard let inference = model?.inference else {
            completion(.failure(PigeonError(code: "Inference model not created", message: nil, details: nil)))
            return
        }

        if enableThinking == true {
            print("[FlutterGemma] Warning: enableThinking=true is not supported on iOS (MediaPipe). " +
                  "Use Android or Desktop with .litertlm models for Gemma 4 thinking mode.")
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let newSession = try InferenceSession(
                    inference: inference,
                    temperature: Float(temperature),
                    randomSeed: Int(randomSeed),
                    topk: Int(topK),
                    topP: topP,
                    loraPath: loraPath,
                    enableVisionModality: enableVisionModality ?? false,
                    enableAudioModality: enableAudioModality ?? false
                )
                DispatchQueue.main.async {
                    self.session = newSession
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func closeSession(completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        session = nil
        completion(.success(()))
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func sizeInTokens(prompt: String, completion: @escaping (Result<Int64, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let tokenCount = try session.sizeInTokens(prompt: prompt)
                DispatchQueue.main.async { completion(.success(Int64(tokenCount))) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func addQueryChunk(prompt: String, completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try session.addQueryChunk(prompt: prompt)
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    // Add method for adding image
    func addImage(imageBytes: FlutterStandardTypedData, completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard let uiImage = UIImage(data: imageBytes.data) else {
                    DispatchQueue.main.async {
                        completion(.failure(PigeonError(code: "Invalid image data", message: "Could not create UIImage from data", details: nil)))
                    }
                    return
                }

                guard let cgImage = uiImage.cgImage else {
                    DispatchQueue.main.async {
                        completion(.failure(PigeonError(code: "Invalid image format", message: "Could not get CGImage from UIImage", details: nil)))
                    }
                    return
                }

                try session.addImage(image: cgImage)

                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    // Add audio input (supported since MediaPipe 0.10.33)
    func addAudio(audioBytes: FlutterStandardTypedData, completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try session.addAudio(audio: audioBytes.data)
                DispatchQueue.main.async {
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func generateResponse(completion: @escaping (Result<String, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let response = try session.generateResponse()
                DispatchQueue.main.async { completion(.success(response)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    @available(iOS 13.0, *)
    func generateResponseAsync(completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        print("[PLUGIN LOG] generateResponseAsync called")
        guard let session = session, let eventSink = eventSink else {
            print("[PLUGIN LOG] Session or eventSink not created")
            completion(.failure(PigeonError(code: "Session or eventSink not created", message: nil, details: nil)))
            return
        }
        
        print("[PLUGIN LOG] Session and eventSink available, starting generation")
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                print("[PLUGIN LOG] Getting async stream from session")
                let stream = try session.generateResponseAsync()
                print("[PLUGIN LOG] Got stream, starting Task")
                Task.detached { [weak self] in
                    guard let self = self else { 
                        print("[PLUGIN LOG] Self is nil in Task")
                        return 
                    }
                    do {
                        print("[PLUGIN LOG] Starting to iterate over stream")
                        var tokenCount = 0
                        for try await token in stream {
                            tokenCount += 1
                            print("[PLUGIN LOG] Got token #\(tokenCount): '\(token)'")
                            DispatchQueue.main.async {
                                print("[PLUGIN LOG] Sending token to Flutter via eventSink")
                                eventSink(["partialResult": token, "done": false])
                                print("[PLUGIN LOG] Token sent to Flutter")
                            }
                        }
                        print("[PLUGIN LOG] Stream finished after \(tokenCount) tokens")
                        DispatchQueue.main.async {
                            print("[PLUGIN LOG] Sending FlutterEndOfEventStream")
                            eventSink(FlutterEndOfEventStream)
                            print("[PLUGIN LOG] FlutterEndOfEventStream sent")
                        }
                    } catch {
                        print("[PLUGIN LOG] Error in stream iteration: \(error)")
                        DispatchQueue.main.async {
                            eventSink(FlutterError(code: "ERROR", message: error.localizedDescription, details: nil))
                        }
                    }
                }
                DispatchQueue.main.async {
                    print("[PLUGIN LOG] Completing with success")
                    completion(.success(()))
                }
            } catch {
                print("[PLUGIN LOG] Error creating stream: \(error)")
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    func stopGeneration(completion: @escaping (Result<Void, any Error>) -> Void) {
        #if canImport(MediaPipeTasksGenAI)

        guard let session = session else {
            completion(.failure(PigeonError(code: "Session not created", message: nil, details: nil)))
            return
        }

        do {
            try session.cancelGeneration()
            completion(.success(()))
        } catch {
            completion(.failure(error))
        }
    
        #else
        completion(.failure(_notAvailableError()))
        #endif
    }

    // MARK: - RAG Methods (iOS Implementation)
    // Embedding features require TensorFlowLite which has no simulator slices.
    // On simulator, all embedding methods return "not available" errors.

    func createEmbeddingModel(modelPath: String, tokenizerPath: String, preferredBackend: PreferredBackend?, completion: @escaping (Result<Void, Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.failure(PigeonError(
            code: "SimulatorNotSupported",
            message: "Embedding models are not supported on iOS simulator",
            details: nil
        )))
        #else
        print("[PLUGIN] Creating embedding model")
        print("[PLUGIN] Model path: \(modelPath)")
        print("[PLUGIN] Tokenizer path: \(tokenizerPath)")
        print("[PLUGIN] Preferred backend: \(String(describing: preferredBackend))")

        let useGPU = preferredBackend == .gpu

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                self.embeddingModel = EmbeddingModel(
                    modelPath: modelPath,
                    tokenizerPath: tokenizerPath,
                    useGPU: useGPU
                )

                try self.embeddingModel?.loadModel()

                DispatchQueue.main.async {
                    print("[PLUGIN] Embedding model created successfully")
                    completion(.success(()))
                }
            } catch {
                DispatchQueue.main.async {
                    print("[PLUGIN] Failed to create embedding model: \(error)")
                    completion(.failure(PigeonError(
                        code: "EmbeddingCreationFailed",
                        message: "Failed to create embedding model: \(error.localizedDescription)",
                        details: nil
                    )))
                }
            }
        }
        #endif
    }
    
    func closeEmbeddingModel(completion: @escaping (Result<Void, Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.success(()))
        #else
        print("[PLUGIN] Closing embedding model")
        DispatchQueue.global(qos: .userInitiated).async {
            self.embeddingModel?.close()
            self.embeddingModel = nil
            DispatchQueue.main.async {
                print("[PLUGIN] Embedding model closed successfully")
                completion(.success(()))
            }
        }
        #endif
    }
    
    func generateEmbeddingFromModel(text: String, completion: @escaping (Result<[Double], Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.failure(PigeonError(code: "SimulatorNotSupported", message: "Embeddings not supported on simulator", details: nil)))
        #else
        guard let embeddingModel = embeddingModel else {
            completion(.failure(PigeonError(code: "EmbeddingModelNotInitialized", message: "Embedding model not initialized.", details: nil)))
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let e = try embeddingModel.generateEmbedding(for: text)
                DispatchQueue.main.async { completion(.success(e.map { Double($0) })) }
            } catch {
                DispatchQueue.main.async { completion(.failure(PigeonError(code: "EmbeddingGenerationFailed", message: error.localizedDescription, details: nil))) }
            }
        }
        #endif
    }

    func generateDocumentEmbeddingFromModel(text: String, completion: @escaping (Result<[Double], Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.failure(PigeonError(code: "SimulatorNotSupported", message: "Embeddings not supported on simulator", details: nil)))
        #else
        guard let embeddingModel = embeddingModel else {
            completion(.failure(PigeonError(code: "EmbeddingModelNotInitialized", message: "Embedding model not initialized.", details: nil)))
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let e = try embeddingModel.generateDocumentEmbedding(for: text)
                DispatchQueue.main.async { completion(.success(e.map { Double($0) })) }
            } catch {
                DispatchQueue.main.async { completion(.failure(PigeonError(code: "DocumentEmbeddingGenerationFailed", message: error.localizedDescription, details: nil))) }
            }
        }
        #endif
    }

    func generateEmbeddingsFromModel(texts: [String], completion: @escaping (Result<[Any?], Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.failure(PigeonError(code: "SimulatorNotSupported", message: "Embeddings not supported on simulator", details: nil)))
        #else
        guard let embeddingModel = embeddingModel else {
            completion(.failure(PigeonError(code: "EmbeddingModelNotInitialized", message: "Embedding model not initialized.", details: nil)))
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                var embeddings: [[Double]] = []
                for text in texts { embeddings.append(try embeddingModel.generateEmbedding(for: text).map { Double($0) }) }
                DispatchQueue.main.async { completion(.success(embeddings as [Any?])) }
            } catch {
                DispatchQueue.main.async { completion(.failure(PigeonError(code: "EmbeddingGenerationFailed", message: error.localizedDescription, details: nil))) }
            }
        }
        #endif
    }

    func getEmbeddingDimension(completion: @escaping (Result<Int64, Error>) -> Void) {
        #if !canImport(TensorFlowLite)
        completion(.failure(PigeonError(code: "SimulatorNotSupported", message: "Embeddings not supported on simulator", details: nil)))
        #else
        guard let embeddingModel = embeddingModel else {
            completion(.failure(PigeonError(code: "EmbeddingModelNotInitialized", message: "Embedding model not initialized.", details: nil)))
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let dim = Int64(try embeddingModel.generateEmbedding(for: "test").count)
                DispatchQueue.main.async { completion(.success(dim)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(PigeonError(code: "EmbeddingDimensionFailed", message: error.localizedDescription, details: nil))) }
            }
        }
        #endif
    }
    
    // MARK: - RAG VectorStore Methods (no-ops: VectorStore is now handled entirely in Dart via sqlite3)

    func initializeVectorStore(databasePath: String, completion: @escaping (Result<Void, Error>) -> Void) {
        completion(.success(()))
    }

    func addDocument(id: String, content: String, embedding: [Double], metadata: String?, completion: @escaping (Result<Void, Error>) -> Void) {
        completion(.success(()))
    }

    func searchSimilar(queryEmbedding: [Double], topK: Int64, threshold: Double, completion: @escaping (Result<[RetrievalResult], Error>) -> Void) {
        completion(.success([]))
    }

    func getVectorStoreStats(completion: @escaping (Result<VectorStoreStats, Error>) -> Void) {
        completion(.success(VectorStoreStats(documentCount: 0, vectorDimension: 0)))
    }

    func clearVectorStore(completion: @escaping (Result<Void, Error>) -> Void) {
        completion(.success(()))
    }

    func closeVectorStore(completion: @escaping (Result<Void, Error>) -> Void) {
        completion(.success(()))
    }

    func getAllDocumentsWithEmbeddings(completion: @escaping (Result<[DocumentWithEmbedding], Error>) -> Void) {
        completion(.success([]))
    }

    func getDocumentsByIds(ids: [String], completion: @escaping (Result<[RetrievalResult], Error>) -> Void) {
        completion(.success([]))
    }

    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }
}