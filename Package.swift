// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "semantic-index-sync-kit",
    // Only platforms this repository's CI actually builds are declared:
    // macOS via `swift build`/`swift test` on macos-15, iOS via the companion
    // demo app's `xcodebuild` job. Nothing is claimed that is not verified.
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
    ]
)
