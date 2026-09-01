#ifndef PhysicsList_hh
#define PhysicsList_hh

#include "G4VModularPhysicsList.hh"

class PhysicsList : public G4VModularPhysicsList {
  public:
    PhysicsList();
    ~PhysicsList() override;

    void ConstructProcess() override;
    void SetCuts() override;
};

#endif  // PhysicsList_hh
