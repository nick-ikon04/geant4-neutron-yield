#include "PrimaryGeneratorAction.hh"

#include "G4Event.hh"
#include "G4Exception.hh"
#include "G4GenericMessenger.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include "RunAction.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>

namespace {
const std::array<G4double, 3> kElectronEnergies = {50.0 * MeV, 100.0 * MeV,
                                                   200.0 * MeV};
const std::array<G4double, 4> kProtonEnergies = {100.0 * MeV, 200.0 * MeV,
                                                 400.0 * MeV, 800.0 * MeV};

bool IsAllowedEnergy(const G4String& particle, G4double energy) {
  if (particle == "proton") {
    for (auto value : kProtonEnergies) {
      if (std::abs(energy - value) < 1e-6 * MeV) {
        return true;
      }
    }
  } else {
    for (auto value : kElectronEnergies) {
      if (std::abs(energy - value) < 1e-6 * MeV) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(),
      fParticleGun(new G4ParticleGun(1)),
      fMessenger(nullptr),
      fParticleName("e-"),
      fBeamEnergy(50.0 * MeV),
      fTargetHalfLength(10.0 * mm) {
  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0., 0., 1.));
  ConfigureMessenger();
  UpdatePrimaryDefinition();
  fParticleGun->SetParticleEnergy(fBeamEnergy);
  fParticleGun->SetParticlePosition(
      G4ThreeVector(0., 0., -fTargetHalfLength - 1.0 * mm));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
  delete fMessenger;
  delete fParticleGun;
}

void PrimaryGeneratorAction::ConfigureMessenger() {
  fMessenger = new G4GenericMessenger(this, "/beam/", "Beam configuration");

  auto& particleCmd = fMessenger->DeclareMethod(
      "particle", &PrimaryGeneratorAction::SetBeamParticle,
      "Select beam particle: e- or proton.");
  particleCmd.SetParameterName("name", false);

  auto& energyCmd = fMessenger->DeclareMethodWithUnit(
      "energy", "MeV", &PrimaryGeneratorAction::SetBeamEnergy,
      "Set kinetic energy of the primary particle.");
  energyCmd.SetParameterName("value", false);
  energyCmd.SetRange("value>0.");

  auto& halfLengthCmd = fMessenger->DeclareMethodWithUnit(
      "targetHalfLength", "mm", &PrimaryGeneratorAction::SetTargetHalfLength,
      "Synchronize source position with the target half-length.");
  halfLengthCmd.SetParameterName("L", false);
  halfLengthCmd.SetRange("L>0.");
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  fParticleGun->SetParticleEnergy(fBeamEnergy);
  fParticleGun->SetParticlePosition(
      G4ThreeVector(0., 0., -fTargetHalfLength - 1.0 * mm));

  auto runActionConst = dynamic_cast<const RunAction*>(
      G4RunManager::GetRunManager()->GetUserRunAction());
  if (runActionConst) {
    auto runAction = const_cast<RunAction*>(runActionConst);
    runAction->RecordPrimary();
  }

  fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGeneratorAction::SetBeamEnergy(G4double energy) {
  if (energy <= 0.) {
    G4Exception("PrimaryGeneratorAction::SetBeamEnergy", "InvalidEnergy",
                JustWarning, "Beam energy must be positive.");
    return;
  }

  fBeamEnergy = energy;
  if (!IsAllowedEnergy(fParticleName, fBeamEnergy)) {
    G4Exception(
        "PrimaryGeneratorAction::SetBeamEnergy", "UnsupportedEnergy",
        JustWarning,
        "Energy does not match the documented discrete values; proceeding "
        "anyway.");
  }
}

void PrimaryGeneratorAction::SetBeamParticle(const G4String& name) {
  auto particle = name;
  std::transform(particle.begin(), particle.end(), particle.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (particle == "e-" || particle == "electron") {
    fParticleName = "e-";
  } else if (particle == "p" || particle == "proton") {
    fParticleName = "proton";
  } else {
    G4Exception("PrimaryGeneratorAction::SetBeamParticle", "InvalidParticle",
                JustWarning,
                ("Unsupported particle '" + name +
                 "'. Supported options: e-, proton.")
                    .c_str());
    return;
  }

  UpdatePrimaryDefinition();

  if (!IsAllowedEnergy(fParticleName, fBeamEnergy)) {
    // Snap to the lowest allowed energy for the selected particle.
    fBeamEnergy = (fParticleName == "proton") ? kProtonEnergies.front()
                                              : kElectronEnergies.front();
  }
}

void PrimaryGeneratorAction::SetTargetHalfLength(G4double halfLength) {
  if (halfLength <= 0.) {
    G4Exception("PrimaryGeneratorAction::SetTargetHalfLength", "InvalidLength",
                JustWarning, "Target half-length must be positive.");
    return;
  }
  fTargetHalfLength = halfLength;
}

void PrimaryGeneratorAction::UpdatePrimaryDefinition() {
  auto particleTable = G4ParticleTable::GetParticleTable();
  auto definition = particleTable->FindParticle(fParticleName);
  if (!definition) {
    G4Exception("PrimaryGeneratorAction::UpdatePrimaryDefinition",
                "ParticleNotFound", FatalException,
                ("Unable to find particle " + fParticleName).c_str());
    return;
  }

  fParticleGun->SetParticleDefinition(definition);

  if (!IsAllowedEnergy(fParticleName, fBeamEnergy)) {
    fBeamEnergy = (fParticleName == "proton") ? kProtonEnergies.front()
                                              : kElectronEnergies.front();
  }
}

G4String PrimaryGeneratorAction::GetBeamTypeLabel() const {
  return (fParticleName == "proton") ? "proton" : "electron";
}
