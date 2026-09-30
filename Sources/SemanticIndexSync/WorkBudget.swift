import Foundation

/// Device conditions the re-index queue has to respect.
///
/// Injected rather than read from `ProcessInfo` so the whole admission policy is
/// testable without a device and without a clock.
public struct DeviceConditions: Sendable, Equatable {
    public var thermalState: ThermalState
    public var isOnExternalPower: Bool
    /// `0.0 ... 1.0`. Clamped at construction; a negative or >1 reading from a
    /// flaky sensor must not invert the policy.
    public let batteryFraction: Double
    public var isLowPowerModeEnabled: Bool

    public init(
        thermalState: ThermalState = .nominal,
        isOnExternalPower: Bool = false,
        batteryFraction: Double = 1.0,
        isLowPowerModeEnabled: Bool = false
    ) {
        self.thermalState = thermalState
        self.isOnExternalPower = isOnExternalPower
        self.batteryFraction = batteryFraction.isNaN ? 0.0 : min(max(batteryFraction, 0.0), 1.0)
        self.isLowPowerModeEnabled = isLowPowerModeEnabled
    }
}

/// Mirrors `ProcessInfo.ThermalState` without importing it, so this module stays
/// buildable and testable on Linux CI.
public enum ThermalState: Int, Sendable, Comparable, CaseIterable, CustomStringConvertible {
    case nominal = 0
    case fair = 1
    case serious = 2
    case critical = 3

    public static func < (lhs: ThermalState, rhs: ThermalState) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public var description: String {
        switch self {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        }
    }
}

/// Why a drain pass was refused.
///
/// Refusals are typed rather than a bare `false` because "the re-index is not
/// progressing" is a question a support engineer will eventually have to answer
/// from a log line.
public enum BudgetRejection: Sendable, Equatable, CustomStringConvertible {
    case thermalCeilingExceeded(observed: ThermalState, ceiling: ThermalState)
    case externalPowerRequired
    case batteryBelowFloor(observed: Double, floor: Double)
    case lowPowerModeActive
    case zeroAllowance

    public var description: String {
        switch self {
        case .thermalCeilingExceeded(let observed, let ceiling):
            return "thermal \(observed) exceeds ceiling \(ceiling)"
        case .externalPowerRequired:
            return "external power required"
        case .batteryBelowFloor(let observed, let floor):
            return String(format: "battery %.0f%% below floor %.0f%%", observed * 100, floor * 100)
        case .lowPowerModeActive:
            return "low power mode active"
        case .zeroAllowance:
            return "budget allows zero chunks"
        }
    }
}

/// How much re-embedding work one background pass may do.
///
/// The batch size is a *budget*, not a target: a pass does at most this much and
/// then returns, leaving the remainder queued. That is what makes the migration
/// resumable across BGTask expirations rather than all-or-nothing.
public struct WorkBudget: Sendable, Equatable {
    public let maxChunksPerPass: Int
    public let thermalCeiling: ThermalState
    public let requiresExternalPower: Bool
    public let batteryFloor: Double
    public let yieldsToLowPowerMode: Bool

    public init(
        maxChunksPerPass: Int,
        thermalCeiling: ThermalState = .fair,
        requiresExternalPower: Bool = false,
        batteryFloor: Double = 0.2,
        yieldsToLowPowerMode: Bool = true
    ) {
        self.maxChunksPerPass = max(0, maxChunksPerPass)
        self.thermalCeiling = thermalCeiling
        self.requiresExternalPower = requiresExternalPower
        self.batteryFloor = batteryFloor.isNaN ? 0.0 : min(max(batteryFloor, 0.0), 1.0)
        self.yieldsToLowPowerMode = yieldsToLowPowerMode
    }

    /// A conservative default for opportunistic background re-indexing.
    public static let background = WorkBudget(maxChunksPerPass: 32)

    /// What the app may use while the user is watching a progress UI: the
    /// screen is on and they are waiting, so a hotter device is acceptable.
    public static let foregroundInteractive = WorkBudget(
        maxChunksPerPass: 128,
        thermalCeiling: .serious,
        requiresExternalPower: false,
        batteryFloor: 0.0,
        yieldsToLowPowerMode: false
    )

    /// Whether a pass may run right now, and if not, why not.
    public func admits(_ conditions: DeviceConditions) -> BudgetRejection? {
        if maxChunksPerPass <= 0 { return .zeroAllowance }
        if conditions.thermalState > thermalCeiling {
            return .thermalCeilingExceeded(observed: conditions.thermalState, ceiling: thermalCeiling)
        }
        if requiresExternalPower && !conditions.isOnExternalPower { return .externalPowerRequired }
        if yieldsToLowPowerMode && conditions.isLowPowerModeEnabled && !conditions.isOnExternalPower {
            return .lowPowerModeActive
        }
        // A charging device is not subject to the battery floor: the floor exists
        // to protect the user's remaining runtime, which charging restores.
        if !conditions.isOnExternalPower && conditions.batteryFraction < batteryFloor {
            return .batteryBelowFloor(observed: conditions.batteryFraction, floor: batteryFloor)
        }
        return nil
    }
}
