import XCTest
@testable import SemanticIndexSync

final class WorkBudgetTests: XCTestCase {

    func testThermalCeilingRejectsWithTheObservedAndAllowedStates() {
        let budget = WorkBudget(maxChunksPerPass: 32, thermalCeiling: .fair)
        let rejection = budget.admits(DeviceConditions(thermalState: .serious))
        XCTAssertEqual(rejection, .thermalCeilingExceeded(observed: .serious, ceiling: .fair))
        XCTAssertNil(budget.admits(DeviceConditions(thermalState: .fair)))
        XCTAssertNil(budget.admits(DeviceConditions(thermalState: .nominal)))
    }

    func testLowPowerModeYieldsUnlessCharging() {
        let budget = WorkBudget(maxChunksPerPass: 8)
        XCTAssertEqual(
            budget.admits(DeviceConditions(isLowPowerModeEnabled: true)),
            .lowPowerModeActive
        )
        XCTAssertNil(
            budget.admits(DeviceConditions(isOnExternalPower: true, isLowPowerModeEnabled: true))
        )
    }

    func testBatteryFloorAppliesOnlyOffCharger() {
        let budget = WorkBudget(maxChunksPerPass: 8, batteryFloor: 0.3)
        XCTAssertEqual(
            budget.admits(DeviceConditions(batteryFraction: 0.1)),
            .batteryBelowFloor(observed: 0.1, floor: 0.3)
        )
        // Charging restores the runtime the floor exists to protect.
        XCTAssertNil(budget.admits(DeviceConditions(isOnExternalPower: true, batteryFraction: 0.1)))
    }

    func testExternalPowerRequirementIsEnforced() {
        let budget = WorkBudget(maxChunksPerPass: 8, requiresExternalPower: true, batteryFloor: 0)
        XCTAssertEqual(budget.admits(DeviceConditions()), .externalPowerRequired)
        XCTAssertNil(budget.admits(DeviceConditions(isOnExternalPower: true)))
    }

    func testZeroAllowanceIsReportedRatherThanSilentlyDoingNothing() {
        XCTAssertEqual(WorkBudget(maxChunksPerPass: 0).admits(DeviceConditions()), .zeroAllowance)
        XCTAssertEqual(WorkBudget(maxChunksPerPass: -5).admits(DeviceConditions()), .zeroAllowance)
    }

    func testOutOfRangeSensorReadingsAreClampedNotTrusted() {
        XCTAssertEqual(DeviceConditions(batteryFraction: 5).batteryFraction, 1.0)
        XCTAssertEqual(DeviceConditions(batteryFraction: -3).batteryFraction, 0.0)
        XCTAssertEqual(DeviceConditions(batteryFraction: .nan).batteryFraction, 0.0)
        XCTAssertEqual(WorkBudget(maxChunksPerPass: 1, batteryFloor: .nan).batteryFloor, 0.0)
        XCTAssertEqual(WorkBudget(maxChunksPerPass: 1, batteryFloor: 9).batteryFloor, 1.0)
    }

    /// The interactive budget is meant to run while the user watches, so it must
    /// not be blocked by the conditions the background budget yields to.
    func testPresetBudgetsDifferInTheWayTheirNamesClaim() {
        let hot = DeviceConditions(thermalState: .serious, batteryFraction: 0.05, isLowPowerModeEnabled: true)
        XCTAssertNotNil(WorkBudget.background.admits(hot))
        XCTAssertNil(WorkBudget.foregroundInteractive.admits(hot))
        XCTAssertGreaterThan(
            WorkBudget.foregroundInteractive.maxChunksPerPass,
            WorkBudget.background.maxChunksPerPass
        )
    }

    /// Asserting `rawValue` ordering would only restate the enum's declaration.
    /// What matters is that the ordering is what `admits` actually consults, for
    /// every state and every ceiling — so the whole ladder is walked.
    func testAdmissionFollowsTheThermalLadderAtEveryCeiling() {
        for ceiling in ThermalState.allCases {
            let budget = WorkBudget(
                maxChunksPerPass: 8,
                thermalCeiling: ceiling,
                requiresExternalPower: false,
                batteryFloor: 0,
                yieldsToLowPowerMode: false
            )
            for observed in ThermalState.allCases {
                let rejection = budget.admits(DeviceConditions(thermalState: observed))
                if observed.rawValue <= ceiling.rawValue {
                    XCTAssertNil(rejection, "\(observed) should be admitted at ceiling \(ceiling)")
                } else {
                    XCTAssertEqual(
                        rejection,
                        .thermalCeilingExceeded(observed: observed, ceiling: ceiling),
                        "\(observed) should be refused at ceiling \(ceiling)"
                    )
                }
            }
        }
    }
}
