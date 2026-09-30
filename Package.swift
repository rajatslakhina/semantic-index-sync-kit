// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "semantic-index-sync-kit",
    // Only platforms this repository's own CI builds are declared: macOS via
    // `swift build`/`swift test` on macos-15, and iOS via the `ios-simulator`
    // job in .github/workflows/ci.yml, which compiles the SwiftUI module for
    // `generic/platform=iOS Simulator`. Live results are on the Actions tab.
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "SemanticIndexSync", targets: ["SemanticIndexSync"]),
        .library(name: "SemanticIndexSyncUI", targets: ["SemanticIndexSyncUI"]),
    ],
    targets: [
        .target(
            name: "SemanticIndexSync",
            swiftSettings: [.enableUpcomingFeature("ExistentialAny")]
        ),
        .target(
            name: "SemanticIndexSyncUI",
            dependencies: ["SemanticIndexSync"],
            swiftSettings: [.enableUpcomingFeature("ExistentialAny")]
        ),
        .testTarget(
            name: "SemanticIndexSyncTests",
            dependencies: ["SemanticIndexSync"]
        ),
        // The workbench's view model is deliberately NOT behind
        // `#if canImport(SwiftUI)` — only the view is — so the demo's own logic
        // is compiled and tested on Linux CI rather than shipping untested.
        .testTarget(
            name: "SemanticIndexSyncUITests",
            dependencies: ["SemanticIndexSyncUI", "SemanticIndexSync"]
        ),
    ]
)
