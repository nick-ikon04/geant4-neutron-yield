#ifndef SteppingAction_hh
#define SteppingAction_hh

#include "G4UserSteppingAction.hh"
#include "globals.hh"

#include <unordered_set>

class DetectorConstruction;
class RunAction;
class G4LogicalVolume;
class G4Step;

class SteppingAction : public G4UserSteppingAction {
  public:
    SteppingAction(const DetectorConstruction* detector, RunAction* runAction);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

  private:
    const DetectorConstruction* fDetector;
    RunAction* fRunAction;

    std::unordered_set<long long> fRecordedTracks;
    G4int fLastEventId;
};

#endif  // SteppingAction_hh
