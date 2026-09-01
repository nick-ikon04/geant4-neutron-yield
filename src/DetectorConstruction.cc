#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4GenericMessenger.hh"
#include "G4Isotope.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Tubs.hh"
#include "G4VisAttributes.hh"

#include <algorithm>
#include <cctype>

#include "PrimaryGeneratorAction.hh"

#include <CLHEP/Units/PhysicalConstants.h>

namespace {
constexpr G4double kWorldHalfSize = 0.5 * m;
}  // namespace

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(),
      fTargetHalfLength(10.0 * mm),
      fTargetRadius(20.0 * mm),
      fUseUranium(true),
      fMaterialLabel("U"),
      fScoringShellThickness(0.5 * mm),
      fTargetLogical(nullptr),
      fScoringLogical(nullptr),
      fMessenger(nullptr) {
  SetupMessenger();
}

DetectorConstruction::~DetectorConstruction() {
  delete fMessenger;
}

void DetectorConstruction::SetupMessenger() {
  fMessenger =
      new G4GenericMessenger(this, "/detector/", "Detector configuration");

  auto& materialCmd = fMessenger->DeclareMethod(
      "material", &DetectorConstruction::SetTargetMaterial,
      "Set target material: natU or W.");
  materialCmd.SetParameterName("name", false);

  auto& useUraniumCmd = fMessenger->DeclareMethod(
      "useUranium", &DetectorConstruction::SetUseUranium,
      "Use natural uranium if true; tungsten otherwise.");
  useUraniumCmd.SetParameterName("flag", false);

  auto& halfLengthCmd = fMessenger->DeclareMethodWithUnit(
      "halfLength", "mm", &DetectorConstruction::SetTargetHalfLength,
      "Set target half-length.");
  halfLengthCmd.SetParameterName("L", false);
  halfLengthCmd.SetRange("L>0.");

  auto& radiusCmd = fMessenger->DeclareMethodWithUnit(
      "radius", "mm", &DetectorConstruction::SetTargetRadius,
      "Set target radius.");
  radiusCmd.SetParameterName("R", false);
  radiusCmd.SetRange("R>0.");

  auto& scoringCmd = fMessenger->DeclareMethodWithUnit(
      "scoringThickness", "mm",
      &DetectorConstruction::SetScoringShellThickness,
      "Set scoring shell thickness.");
  scoringCmd.SetParameterName("t", false);
  scoringCmd.SetRange("t>0.");
}

G4VPhysicalVolume* DetectorConstruction::Construct() {
  fTargetLogical = nullptr;
  fScoringLogical = nullptr;

  auto nist = G4NistManager::Instance();
  auto vacuum = nist->FindOrBuildMaterial("G4_Galactic");

  auto worldSolid =
      new G4Box("WorldSolid", kWorldHalfSize, kWorldHalfSize, kWorldHalfSize);
  auto worldLogical =
      new G4LogicalVolume(worldSolid, vacuum, "WorldLogical");
  worldLogical->SetVisAttributes(G4VisAttributes::GetInvisible());

  auto worldPhysical = new G4PVPlacement(nullptr, {}, worldLogical,
                                         "WorldPhysical", nullptr, false, 0,
                                         true);

  // Target geometry
  auto targetSolid = new G4Tubs("TargetSolid", 0., fTargetRadius,
                                fTargetHalfLength, 0., CLHEP::twopi);
  auto targetMaterial = CreateTargetMaterial();
  fTargetLogical =
      new G4LogicalVolume(targetSolid, targetMaterial, "TargetLogical");
  auto targetPhys = new G4PVPlacement(
      nullptr, {}, fTargetLogical, "TargetPhysical", worldLogical, false, 0,
      true);
  (void)targetPhys;

  // Scoring shell immediately outside the target.
  auto shellOuterRadius = fTargetRadius + fScoringShellThickness;
  auto shellSolid = new G4Tubs("ScoringShellSolid", fTargetRadius,
                               shellOuterRadius, fTargetHalfLength, 0.,
                               CLHEP::twopi);
  fScoringLogical =
      new G4LogicalVolume(shellSolid, vacuum, "ScoringShellLogical");
  auto shellPhys = new G4PVPlacement(nullptr, {}, fScoringLogical,
                                     "ScoringShellPhysical", worldLogical,
                                     false, 0, true);
  (void)shellPhys;

  return worldPhysical;
}

void DetectorConstruction::SetTargetHalfLength(G4double halfLength) {
  if (halfLength <= 0.) {
    G4Exception("DetectorConstruction::SetTargetHalfLength", "InvalidLength",
                JustWarning, "Target half-length must be positive.");
    return;
  }
  fTargetHalfLength = halfLength;

  fTargetLogical = nullptr;
  fScoringLogical = nullptr;

  auto runManager = G4RunManager::GetRunManager();
  if (runManager) {
    runManager->GeometryHasBeenModified();
    auto generatorConst = dynamic_cast<const PrimaryGeneratorAction*>(
        runManager->GetUserPrimaryGeneratorAction());
    if (generatorConst) {
      auto generator =
          const_cast<PrimaryGeneratorAction*>(generatorConst);
      generator->SetTargetHalfLength(fTargetHalfLength);
    }
  }
}

void DetectorConstruction::SetTargetRadius(G4double radius) {
  if (radius <= 0.) {
    G4Exception("DetectorConstruction::SetTargetRadius", "InvalidRadius",
                JustWarning, "Target radius must be positive.");
    return;
  }
  fTargetRadius = radius;

  fTargetLogical = nullptr;
  fScoringLogical = nullptr;

  auto runManager = G4RunManager::GetRunManager();
  if (runManager) {
    runManager->GeometryHasBeenModified();
  }
}

void DetectorConstruction::SetScoringShellThickness(G4double thickness) {
  if (thickness <= 0.) {
    G4Exception("DetectorConstruction::SetScoringShellThickness",
                "InvalidThickness", JustWarning,
                "Scoring shell thickness must be positive.");
    return;
  }
  fScoringShellThickness = thickness;

  fTargetLogical = nullptr;
  fScoringLogical = nullptr;

  auto runManager = G4RunManager::GetRunManager();
  if (runManager) {
    runManager->GeometryHasBeenModified();
  }
}

void DetectorConstruction::SetUseUranium(G4bool useUranium) {
  fUseUranium = useUranium;
  fMaterialLabel = useUranium ? "U" : "W";

  fTargetLogical = nullptr;
  fScoringLogical = nullptr;

  auto runManager = G4RunManager::GetRunManager();
  if (runManager) {
    runManager->GeometryHasBeenModified();
  }
}

void DetectorConstruction::SetTargetMaterial(const G4String& materialName) {
  auto name = materialName;
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (name == "natu" || name == "nat_u" || name == "uranium" ||
      name == "u" || name == "u238") {
    SetUseUranium(true);
    fMaterialLabel = "U";
  } else if (name == "w" || name == "tungsten") {
    SetUseUranium(false);
    fMaterialLabel = "W";
  } else {
    G4Exception("DetectorConstruction::SetTargetMaterial", "InvalidMaterial",
                JustWarning,
                ("Unknown material '" + materialName +
                 "'. Expected natU or W. Keeping previous choice.")
                    .c_str());
  }
}

G4String DetectorConstruction::GetTargetMaterialLabel() const {
  return fMaterialLabel;
}

G4Material* DetectorConstruction::CreateTargetMaterial() {
  return fUseUranium ? BuildNatUranium() : BuildTungsten();
}

G4Material* DetectorConstruction::BuildNatUranium() {
  auto material = G4Material::GetMaterial("natUranium", false);
  if (material) {
    return material;
  }

  // Natural-uranium atom fractions. U-234 is retained instead of rounding the
  // material to a two-isotope mixture.
  auto U234 =
      new G4Isotope("U234", 92, 234, 234.0409523 * g / mole);
  auto U235 =
      new G4Isotope("U235", 92, 235, 235.0439299 * g / mole);
  auto U238 =
      new G4Isotope("U238", 92, 238, 238.05078826 * g / mole);

  auto natUElement = new G4Element("NatUraniumElement", "U", 3);
  natUElement->AddIsotope(U234, 0.000054);
  natUElement->AddIsotope(U235, 0.007204);
  natUElement->AddIsotope(U238, 0.992742);

  material =
      new G4Material("natUranium", 19.0 * g / cm3, 1, kStateSolid);
  material->AddElement(natUElement, 1.0);

  return material;
}

G4Material* DetectorConstruction::BuildTungsten() {
  auto material = G4Material::GetMaterial("TungstenTarget", false);
  if (material) {
    return material;
  }

  auto nist = G4NistManager::Instance();
  auto elementW = nist->FindOrBuildElement("W");

  material =
      new G4Material("TungstenTarget", 19.3 * g / cm3, 1, kStateSolid);
  material->AddElement(elementW, 1.0);

  return material;
}
