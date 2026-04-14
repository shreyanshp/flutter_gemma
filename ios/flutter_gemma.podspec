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
NOTE: This is a slim fork that removes all native iOS dependencies (MediaPipe, TFLite)
to work around Xcode 26 linker bugs. AI inference is handled on Android/macOS; on iOS
the plugin compiles but inference methods return errors gracefully.
                       DESC
  s.homepage         = 'https://github.com/shreyanshp/flutter_gemma'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Flutter Berlin' => 'flutter@flutterberlin.dev' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/*.swift'
  s.dependency 'Flutter'
  # MediaPipe and TFLite dependencies removed due to Xcode 26 linker
  # dylibToOrdinal crash with static XCFrameworks. All inference code
  # is guarded with #if canImport() so the plugin compiles as a no-op.
  s.platform = :ios, '16.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS[sdk=iphonesimulator*]' => ''
  }
  s.swift_version = '5.0'
end
