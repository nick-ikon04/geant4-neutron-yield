#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"

#include "G4RunManagerFactory.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"

int main(int argc, char** argv) {
  auto* runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly);

  auto* detector = new DetectorConstruction();
  runManager->SetUserInitialization(detector);
  runManager->SetUserInitialization(new PhysicsList());
  runManager->SetUserInitialization(new ActionInitialization(detector));

  auto* uiManager = G4UImanager::GetUIpointer();

  if (argc > 1) {
    G4String macroFile = argv[1];
    uiManager->ApplyCommand("/control/execute " + macroFile);
  } else {
    auto* ui = new G4UIExecutive(argc, argv);
    auto* visManager = new G4VisExecutive();
    visManager->Initialize();

    runManager->Initialize();
    ui->SessionStart();

    delete visManager;
    delete ui;
  }

  delete runManager;
  return 0;
}
