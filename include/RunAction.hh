#ifndef RunAction_hh
#define RunAction_hh

#include "G4ThreeVector.hh"
#include "G4UserRunAction.hh"

#include "globals.hh"

#include <string>

class G4Run;
class G4GenericMessenger;
class DetectorConstruction;
class PrimaryGeneratorAction;

class RunAction : public G4UserRunAction {
  public:
    RunAction(const DetectorConstruction* detector,
              const PrimaryGeneratorAction* primary);
    ~RunAction() override;

    void BeginOfRunAction(const G4Run* run) override;
    void EndOfRunAction(const G4Run* run) override;

    void RecordPrimary();
    void RecordNeutronEscape(G4double energyMeV, G4double cosTheta,
                             const G4ThreeVector& exitPosition);
    void SetFileType(const G4String& fileType);

    G4long GetPrimaryCount() const { return fNprim; }
    G4long GetNeutronEscapeCount() const { return fNneutronEscaped; }
    const G4String& GetFileName() const { return fFileName; }
    const G4String& GetFileType() const { return fFileType; }

  private:
    void ConfigureMessenger();
    void ConfigureOptionalOutputs();
    void WriteRunMetadata(const std::string& analysisBasePath, G4double yield,
                          G4double sigma) const;
    void EnsureOutputDirectoryExists(const std::string& basePath) const;

    const DetectorConstruction* fDetector;
    const PrimaryGeneratorAction* fPrimary;
    G4long fNprim;
    G4long fNneutronEscaped;
    G4String fFileName;
    G4String fFileType;
    G4int fNeutronEnergyH1Id;
    G4int fNeutronCosH1Id;
    G4int fNeutronNtupleId;

    G4GenericMessenger* fMessenger;
    G4bool fEnableAngularHistogram;
    G4bool fEnableEscapedNtuple;
};

#endif  // RunAction_hh
