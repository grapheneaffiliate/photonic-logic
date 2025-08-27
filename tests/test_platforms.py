import pytest

from plogic.materials import PlatformDB, compute_gamma_from_params


def test_db_has_three_platforms():
    pdb = PlatformDB()
    keys = set(pdb.keys())
    assert {"Si", "SiN", "AlGaAs"} <= keys


def test_platform_loading():
    pdb = PlatformDB()

    # Test Si platform
    si = pdb.get("Si")
    assert si.name == "Silicon"
    assert si.nonlinear.n2_m2_per_W == 4.5e-18
    assert si.flags.cmos_compatible is True
    assert si.flags.tpa_present_at_1550 is True

    # Test SiN platform
    sin = pdb.get("SiN")
    assert sin.name == "Silicon Nitride"
    assert sin.nonlinear.n2_m2_per_W == 2.4e-19
    assert sin.flags.tpa_present_at_1550 is False

    # Test AlGaAs platform
    algaas = pdb.get("AlGaAs")
    assert algaas.name == "Aluminum Gallium Arsenide"
    assert algaas.nonlinear.n2_m2_per_W == 1.5e-17
    assert algaas.flags.cmos_compatible is False


def test_platform_not_found():
    pdb = PlatformDB()
    with pytest.raises(KeyError, match="Unknown platform 'InvalidPlatform'"):
        pdb.get("InvalidPlatform")


def test_gamma_computation():
    # Test gamma calculation from n2, wavelength, and Aeff
    n2 = 1e-18
    wavelength_nm = 1550.0
    Aeff_um2 = 0.6

    gamma = compute_gamma_from_params(n2, wavelength_nm, Aeff_um2)
    assert gamma > 0

    # Larger Aeff should give smaller gamma
    gamma_large = compute_gamma_from_params(n2, wavelength_nm, 2.0)
    assert gamma > gamma_large


def test_platform_gamma_calculation():
    pdb = PlatformDB()
    si = pdb.get("Si")

    # Test gamma calculation
    gamma = si.gamma_Winv_m(wavelength_nm=1550.0, Aeff_um2=0.5)
    assert gamma > 0

    # Test with defaults
    gamma_default = si.gamma_Winv_m()
    assert gamma_default > 0


def test_platform_loss_conversion():
    pdb = PlatformDB()
    si = pdb.get("Si")

    # Test loss conversion from dB/cm to alpha/m
    alpha = si.loss_alpha_m()
    assert alpha > 0

    # SiN should have lower loss than Si
    sin = pdb.get("SiN")
    alpha_sin = sin.loss_alpha_m()
    assert alpha_sin < alpha


def test_q_factor_validation():
    pdb = PlatformDB()
    si = pdb.get("Si")

    # Reasonable Q should pass
    warn = si.validate_reasonable_Q(5e5)
    assert warn is None

    # Unreasonably high Q should warn
    warn = si.validate_reasonable_Q(5e6)
    assert warn is not None
    assert "far exceeds" in warn

    # None should pass
    warn = si.validate_reasonable_Q(None)
    assert warn is None


def test_platform_thermal_parameters():
    pdb = PlatformDB()

    # Test thermal parameters are loaded correctly
    si = pdb.get("Si")
    assert si.thermal.dn_dT_per_K == 1.8e-4
    assert si.thermal.tau_thermal_ns == 100.0

    sin = pdb.get("SiN")
    assert sin.thermal.dn_dT_per_K == 2.5e-5
    assert sin.thermal.tau_thermal_ns == 200.0

    algaas = pdb.get("AlGaAs")
    assert algaas.thermal.dn_dT_per_K == 3.0e-4
    assert algaas.thermal.tau_thermal_ns == 60.0


def test_platform_fabrication_parameters():
    pdb = PlatformDB()

    # Test fabrication parameters
    si = pdb.get("Si")
    assert si.fabrication.Q_max == 1e6
    assert si.fabrication.loss_dB_per_cm == 0.1

    sin = pdb.get("SiN")
    assert sin.fabrication.Q_max == 1e7
    assert sin.fabrication.loss_dB_per_cm == 0.01

    algaas = pdb.get("AlGaAs")
    assert algaas.fabrication.Q_max == 5e5
    assert algaas.fabrication.loss_dB_per_cm == 0.5
