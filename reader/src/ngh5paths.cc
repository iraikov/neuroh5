#include "ngh5paths.hh"

using namespace std;

namespace ngh5
{
  const string H5PathNames::CONN = "Connectivity";

  const string H5PathNames::DST_BLK_PTR =
    "/Connectivity/Destination Block Pointer";

  const string H5PathNames::DST_IDX = "/Connectivity/Destination Index";

  const string H5PathNames::DST_PTR = "/Connectivity/Destination Pointer";

  const string H5PathNames::DST_POP = "/Destination Population";

  const string H5PathNames::SRC_IDX = "/Connectivity/Source Index";

  const string H5PathNames::SRC_POP = "/Destination Population";

  const string H5PathNames::POP_COMB = "Valid population projections";

  const string H5PathNames::POP_RNG = "Populations";
}
