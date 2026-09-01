#include "PhysicsList.hh"

#include "G4DecayPhysics.hh"
#include "G4EmExtraPhysics.hh"
#include "G4EmParameters.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4HadronPhysicsFTFP_BERT_HP.hh"
#include "G4IonPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4StoppingPhysics.hh"

PhysicsList::PhysicsList() : G4VModularPhysicsList() {
  SetVerboseLevel(1);

  RegisterPhysics(new G4EmStandardPhysics_option4());

  auto emExtra = new G4EmExtraPhysics();
  emExtra->Synch(true);
  emExtra->GammaNuclear(true);
  emExtra->MuonNuclear(true);
  RegisterPhysics(emExtra);

  RegisterPhysics(new G4DecayPhysics());
  RegisterPhysics(new G4RadioactiveDecayPhysics());

  RegisterPhysics(new G4HadronElasticPhysicsHP());
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT_HP());
  RegisterPhysics(new G4StoppingPhysics());
  RegisterPhysics(new G4IonPhysics());
  RegisterPhysics(new G4NeutronTrackingCut());

  auto emParams = G4EmParameters::Instance();
  emParams->SetMscRangeFactor(0.02);
  emParams->SetFluo(true);
  emParams->SetAugerCascade(true);
  emParams->SetAuger(true);
  emParams->SetUseMottCorrection(true);
  emParams->SetVerbose(0);
}

PhysicsList::~PhysicsList() = default;

void PhysicsList::ConstructProcess() {
  G4VModularPhysicsList::ConstructProcess();
}

void PhysicsList::SetCuts() {
  SetCutsWithDefault();
}
