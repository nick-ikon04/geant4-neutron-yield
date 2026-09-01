#include "RunAction.hh"

#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4AnalysisManager.hh"
#include "G4GenericMessenger.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

namespace {
std::string StripExtension(const std::string& name, const std::string& extension) {
  if (extension.empty()) {
    return name;
  }
  const std::string suffix = "." + extension;
  if (name.size() > suffix.size() &&
      name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
    return name.substr(0, name.size() - suffix.size());
  }
  return name;
}
}  // namespace

RunAction::RunAction(const DetectorConstruction* detector,
                     const PrimaryGeneratorAction* primary)
    : G4UserRunAction(),
      fDetector(detector),
      fPrimary(primary),
      fNprim(0),
      fNneutronEscaped(0),
      fFileName("neutron_yield"),
      fFileType("csv"),
      fNeutronEnergyH1Id(-1),
      fNeutronCosH1Id(-1),
      fNeutronNtupleId(-1),
      fMessenger(nullptr),
      fEnableAngularHistogram(false),
      fEnableEscapedNtuple(false) {
  ConfigureMessenger();

  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->SetDefaultFileType(fFileType);
  analysisManager->SetFileName(fFileName);
  analysisManager->SetVerboseLevel(0);
  fNeutronEnergyH1Id = analysisManager->CreateH1(
      "hNeutronEnergy", "Escaped neutron kinetic energy", 400, 0.0, 200.0);
}

RunAction::~RunAction() {
  delete fMessenger;
}

void RunAction::ConfigureMessenger() {
  fMessenger = new G4GenericMessenger(this, "/analysis/", "Analysis control");

  auto& fileCmd = fMessenger->DeclareProperty(
      "fileName", fFileName,
      "Output file name (without extension) for histograms.");
  fileCmd.SetParameterName("name", false);

  auto& typeCmd = fMessenger->DeclareMethod(
      "fileType", &RunAction::SetFileType,
      "Set output file type (csv or root).");
  typeCmd.SetParameterName("type", false);

  auto& angularCmd = fMessenger->DeclareProperty(
      "enableAngularHistogram", fEnableAngularHistogram,
      "Enable writing the escaped neutron angular distribution histogram.");
  angularCmd.SetParameterName("enable", false);

  auto& ntupleCmd = fMessenger->DeclareProperty(
      "enableEscapedNtuple", fEnableEscapedNtuple,
      "Enable writing the escaped neutron ntuple.");
  ntupleCmd.SetParameterName("enable", false);
}

void RunAction::BeginOfRunAction(const G4Run* /*run*/) {
  fNprim = 0;
  fNneutronEscaped = 0;

  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->SetDefaultFileType(fFileType);
  analysisManager->SetFileName(fFileName);
  ConfigureOptionalOutputs();
  EnsureOutputDirectoryExists(fFileName.c_str());
  analysisManager->OpenFile();
}

void RunAction::EndOfRunAction(const G4Run* run) {
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->Write();
  analysisManager->CloseFile();

  G4double yield = (fNprim > 0) ? static_cast<G4double>(fNneutronEscaped) /
                                      static_cast<G4double>(fNprim)
                                : 0.0;
  G4double sigma =
      (fNprim > 0)
          ? std::sqrt(static_cast<G4double>(fNneutronEscaped)) /
                static_cast<G4double>(fNprim)
          : 0.0;

  const std::string managerName = analysisManager->GetFileName();
  const std::string defaultExt = analysisManager->GetDefaultFileType();
  const std::string baseName = StripExtension(managerName, defaultExt);
  const std::string outputPath = baseName + "." + defaultExt;

  G4cout << "=== Run summary =====================================" << G4endl;
  G4cout << " Events processed: " << run->GetNumberOfEvent() << G4endl;
  G4cout << " Primary particles: " << fNprim << G4endl;
  G4cout << " Escaped neutrons : " << fNneutronEscaped << G4endl;
  G4cout << " Yield (neutrons / primary): " << yield << G4endl;
  G4cout << " Output written to: " << outputPath << G4endl;
  G4cout << "=====================================================" << G4endl;

  WriteRunMetadata(baseName, yield, sigma);
}

void RunAction::RecordPrimary() {
  ++fNprim;
}

void RunAction::RecordNeutronEscape(G4double energyMeV, G4double cosTheta,
                                    const G4ThreeVector& exitPosition) {
  ++fNneutronEscaped;
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->FillH1(fNeutronEnergyH1Id, energyMeV);
  if (fNeutronCosH1Id >= 0) {
    analysisManager->FillH1(fNeutronCosH1Id, cosTheta);
  }

  if (fNeutronNtupleId >= 0) {
    analysisManager->FillNtupleDColumn(fNeutronNtupleId, 0, energyMeV);
    analysisManager->FillNtupleDColumn(fNeutronNtupleId, 1, cosTheta);
    analysisManager->FillNtupleDColumn(fNeutronNtupleId, 2,
                                       exitPosition.x() / mm);
    analysisManager->FillNtupleDColumn(fNeutronNtupleId, 3,
                                       exitPosition.y() / mm);
    analysisManager->FillNtupleDColumn(fNeutronNtupleId, 4,
                                       exitPosition.z() / mm);
    analysisManager->AddNtupleRow(fNeutronNtupleId);
  }
}

void RunAction::SetFileType(const G4String& fileType) {
  auto type = fileType;
  G4String lower = type;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (lower == "csv" || lower == "root") {
    fFileType = lower;
  } else {
    G4Exception("RunAction::SetFileType", "InvalidFileType", JustWarning,
                ("Unsupported file type '" + fileType +
                 "'. Supported types: csv, root. Keeping " + fFileType + ".")
                    .c_str());
  }
}

void RunAction::EnsureOutputDirectoryExists(const std::string& basePath) const {
  std::filesystem::path path(basePath);
  auto parent = path.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      G4Exception("RunAction::EnsureOutputDirectoryExists", "DirectoryError",
                  JustWarning,
                  ("Failed to create directory '" + parent.string() +
                   "': " + ec.message())
                      .c_str());
    }
  }
}

void RunAction::WriteRunMetadata(const std::string& analysisBasePath,
                                 G4double yield, G4double sigma) const {
  std::string base = StripExtension(analysisBasePath, std::string(fFileType));
  const std::string extension = fFileType.c_str();
  const std::string beamType =
      fPrimary ? fPrimary->GetBeamTypeLabel().c_str() : "unknown";
  const G4double beamEnergy =
      fPrimary ? fPrimary->GetBeamEnergy() / MeV : 0.0;
  const std::string material =
      fDetector ? fDetector->GetTargetMaterialLabel().c_str() : "unknown";
  const G4double halfLength =
      fDetector ? fDetector->GetTargetHalfLength() / mm : 0.0;
  const G4double radius =
      fDetector ? fDetector->GetTargetRadius() / mm : 0.0;
  const G4double scoringThickness =
      fDetector ? fDetector->GetScoringShellThickness() / mm : 0.0;

  std::filesystem::path metadataPath(base + "_metadata.json");

  try {
    auto parent = metadataPath.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }

    std::ofstream ofs(metadataPath);
    if (!ofs) {
      std::ostringstream msg;
      msg << "Failed to open metadata file for writing: "
          << metadataPath.string();
      G4Exception("RunAction::WriteRunMetadata", "MetadataIOError",
                  JustWarning, msg.str().c_str());
      return;
    }

    ofs << std::fixed << std::setprecision(6);
    ofs << "{\n";
    ofs << "  \"beam_type\": \"" << beamType << "\",\n";
    ofs << "  \"beam_energy_MeV\": " << beamEnergy << ",\n";
    ofs << "  \"target_material\": \"" << material << "\",\n";
    ofs << "  \"target_half_length_mm\": " << halfLength << ",\n";
    ofs << "  \"target_radius_mm\": " << radius << ",\n";
    ofs << "  \"scoring_shell_thickness_mm\": " << scoringThickness << ",\n";
    ofs << "  \"N_primary\": " << fNprim << ",\n";
    ofs << "  \"N_neutrons\": " << fNneutronEscaped << ",\n";
    ofs << "  \"yield_per_primary\": " << yield << ",\n";
    ofs << "  \"sigma_yield\": " << sigma << ",\n";
    ofs << "  \"output\": {\n";
    ofs << "    \"base\": \"" << base << "\",\n";
    ofs << "    \"file_type\": \"" << extension << "\",\n";
    ofs << "    \"h1_energy\": \"" << base << "_h1_hNeutronEnergy." << extension
        << "\"";
    if (fNeutronCosH1Id >= 0) {
      ofs << ",\n    \"h1_cosTheta\": \"" << base << "_h1_hNeutronCosTheta."
          << extension << "\"";
    }
    if (fNeutronNtupleId >= 0) {
      ofs << ",\n    \"ntuple\": \"" << base << "_nt_escaped_neutrons."
          << extension << "\"";
    }
    ofs << "\n";
    ofs << "  }\n";
    ofs << "}\n";
  } catch (const std::exception& ex) {
    G4Exception("RunAction::WriteRunMetadata", "MetadataException", JustWarning,
                ("Failed to write metadata: " + std::string(ex.what())).c_str());
  }
}

void RunAction::ConfigureOptionalOutputs() {
  auto analysisManager = G4AnalysisManager::Instance();
  if (fEnableAngularHistogram && fNeutronCosH1Id < 0) {
    fNeutronCosH1Id = analysisManager->CreateH1(
        "hNeutronCosTheta", "Escaped neutron cos(theta)", 200, -1.0, 1.0);
  }
  if (fEnableEscapedNtuple && fNeutronNtupleId < 0) {
    fNeutronNtupleId = analysisManager->CreateNtuple(
        "escaped_neutrons", "Escaping neutron observables");
    analysisManager->CreateNtupleDColumn(fNeutronNtupleId, "energy_MeV");
    analysisManager->CreateNtupleDColumn(fNeutronNtupleId, "cosTheta");
    analysisManager->CreateNtupleDColumn(fNeutronNtupleId, "x_exit_mm");
    analysisManager->CreateNtupleDColumn(fNeutronNtupleId, "y_exit_mm");
    analysisManager->CreateNtupleDColumn(fNeutronNtupleId, "z_exit_mm");
    analysisManager->FinishNtuple(fNeutronNtupleId);
  }
}
