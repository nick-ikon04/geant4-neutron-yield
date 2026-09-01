// DetectorConstruction.hh
// Defines simulation geometry and configurable target material/size.

#ifndef DetectorConstruction_hh
#define DetectorConstruction_hh

#include "G4VUserDetectorConstruction.hh"

#include "globals.hh"

class G4LogicalVolume;
class G4Material;
class G4GenericMessenger;
class G4VPhysicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction {
  public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;

    void SetTargetHalfLength(G4double halfLength);
    void SetUseUranium(G4bool useUranium);
    void SetTargetMaterial(const G4String& materialName);
    void SetTargetRadius(G4double radius);
    void SetScoringShellThickness(G4double thickness);

    G4double GetTargetHalfLength() const { return fTargetHalfLength; }
    G4double GetTargetRadius() const { return fTargetRadius; }
    G4double GetScoringShellThickness() const { return fScoringShellThickness; }
    G4bool GetUseUranium() const { return fUseUranium; }
    G4String GetTargetMaterialLabel() const;
    G4LogicalVolume* GetTargetLogical() const { return fTargetLogical; }
    G4LogicalVolume* GetScoringLogical() const { return fScoringLogical; }

    static G4Material* BuildNatUranium();
    static G4Material* BuildTungsten();

  private:
    G4Material* CreateTargetMaterial();
    void SetupMessenger();

    G4double fTargetHalfLength;
    G4double fTargetRadius;
    G4bool fUseUranium;
    G4String fMaterialLabel;
    G4double fScoringShellThickness;

    G4LogicalVolume* fTargetLogical;
    G4LogicalVolume* fScoringLogical;

    G4GenericMessenger* fMessenger;
};

#endif  // DetectorConstruction_hh
