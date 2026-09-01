#ifndef PrimaryGeneratorAction_hh
#define PrimaryGeneratorAction_hh

#include "G4VUserPrimaryGeneratorAction.hh"

#include "globals.hh"

class G4ParticleGun;
class G4GenericMessenger;
class G4Event;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
  public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    void SetBeamEnergy(G4double energy);
    void SetBeamParticle(const G4String& name);
    void SetTargetHalfLength(G4double halfLength);

    G4double GetBeamEnergy() const { return fBeamEnergy; }
    const G4String& GetBeamParticle() const { return fParticleName; }
    G4String GetBeamTypeLabel() const;

  private:
    void ConfigureMessenger();
    void UpdatePrimaryDefinition();

    G4ParticleGun* fParticleGun;
    G4GenericMessenger* fMessenger;

    G4String fParticleName;
    G4double fBeamEnergy;
    G4double fTargetHalfLength;
};

#endif  // PrimaryGeneratorAction_hh
