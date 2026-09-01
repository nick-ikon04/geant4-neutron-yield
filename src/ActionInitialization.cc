#include "ActionInitialization.hh"

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

ActionInitialization::ActionInitialization(DetectorConstruction* detector)
    : G4VUserActionInitialization(), fDetector(detector) {}

void ActionInitialization::BuildForMaster() const {
  SetUserAction(new RunAction(fDetector, nullptr));
}

void ActionInitialization::Build() const {
  auto primary = new PrimaryGeneratorAction();
  if (fDetector) {
    primary->SetTargetHalfLength(fDetector->GetTargetHalfLength());
  }
  SetUserAction(primary);

  auto runAction = new RunAction(fDetector, primary);
  SetUserAction(runAction);

  SetUserAction(new SteppingAction(fDetector, runAction));
}
