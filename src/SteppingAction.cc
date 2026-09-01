#include "SteppingAction.hh"

#include "DetectorConstruction.hh"
#include "RunAction.hh"

#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4Track.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"

SteppingAction::SteppingAction(const DetectorConstruction* detector,
                               RunAction* runAction)
    : G4UserSteppingAction(),
      fDetector(detector),
      fRunAction(runAction),
      fRecordedTracks(),
      fLastEventId(-1) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {
  if (!step || !fRunAction) {
    return;
  }

  auto track = step->GetTrack();
  if (!track) {
    return;
  }

  auto definition = track->GetDefinition();
  if (!definition || definition->GetParticleName() != "neutron") {
    return;
  }

  auto postPoint = step->GetPostStepPoint();
  if (!postPoint || postPoint->GetStepStatus() != fGeomBoundary) {
    return;  // We only care about boundary crossings.
  }

  auto prePoint = step->GetPreStepPoint();
  if (!prePoint) {
    return;
  }

  auto preVolumeHandle = prePoint->GetTouchableHandle();
  auto postVolumeHandle = postPoint->GetTouchableHandle();
  if (!preVolumeHandle) {
    return;
  }

  auto preVolume = preVolumeHandle->GetVolume();
  auto postVolume = postVolumeHandle ? postVolumeHandle->GetVolume() : nullptr;

  auto preLogical = preVolume ? preVolume->GetLogicalVolume() : nullptr;
  auto postLogical = postVolume ? postVolume->GetLogicalVolume() : nullptr;

  if (!preLogical) {
    return;
  }

  // Count the first escape through any target surface. The previous radial-
  // shell-only condition missed neutrons leaving through either end cap.
  if (preLogical->GetName() != "TargetLogical" ||
      (postLogical && postLogical->GetName() == "TargetLogical")) {
    return;
  }

  auto runManager = G4RunManager::GetRunManager();
  auto currentEvent =
      runManager ? runManager->GetCurrentEvent() : nullptr;
  auto eventId = currentEvent ? currentEvent->GetEventID() : -1;
  if (eventId != fLastEventId) {
    fRecordedTracks.clear();
    fLastEventId = eventId;
  }
  auto trackId = track->GetTrackID();

  long long key =
      (static_cast<long long>(eventId) << 32) |
      static_cast<unsigned int>(trackId);
  if (!fRecordedTracks.insert(key).second) {
    return;  // Already recorded this track crossing.
  }

  const G4double energyMeV = track->GetKineticEnergy() / MeV;
  const auto momentumDir = track->GetMomentumDirection();
  const G4double cosTheta = momentumDir.z();
  const auto exitPosition = postPoint->GetPosition();

  fRunAction->RecordNeutronEscape(energyMeV, cosTheta, exitPosition);
}
