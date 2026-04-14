#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint flutter_gemma.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'flutter_gemma'
  s.version          = '0.13.2'
  s.summary          = 'Flutter plugin for running Gemma AI models locally.'
  s.description      = <<-DESC
The plugin allows running the Gemma AI model locally on a device from a Flutter application.
Fork: Fixes Xcode 26 linker crash by removing -force_load and TFLite deps.
MediaPipe GenAI is included for on-device LLM inference.
                       DESC
  s.homepage         = 'https://github.com/shreyanshp/flutter_gemma'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Flutter Berlin' => 'flutter@flutterberlin.dev' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/*.swift'
  s.dependency 'Flutter'
  s.dependency 'MediaPipeTasksGenAI', '= 0.10.33'
  s.dependency 'MediaPipeTasksGenAIC', '= 0.10.33'
  # TFLite deps removed — only needed for embedding models, not chat inference.
  # Embedding code is guarded with #if canImport(TensorFlowLite).
  s.platform = :ios, '16.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    # NO -force_load — this triggers the Xcode 26 ld_prime dylibToOrdinal crash.
    # MediaPipe symbols are linked normally via the framework dependency.
    # The standalone .a files are NOT force-loaded.
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' => ''
  }
  s.swift_version = '5.0'
end
