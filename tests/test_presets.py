"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink rastet ein."""

import elw_constants as C
import elw_evaluation as ev
import elw_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_ranges():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.options is not None:
                assert preset[key] in spec.options, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = C.PRESETS[next(iter(C.PRESETS))]
    assert ev.Settings(first["welt"], first["tau"], first["n"], first["p"], first["seed"]) == ev.Settings()
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == first[key], key


def test_setting_specs_cover_every_preset_key_and_every_default_tree_answer():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(C.DEFAULT_TREE) <= set(P.PRESET_KEYS)


def test_url_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_snap_rounds_to_levels_and_ignores_unusable_names():
    n_spec = P.SETTING_SPECS["n_select"]
    assert P.snap(n_spec, 250) == 100 and P.snap(n_spec, 700) == 1000 and P.snap(n_spec, 10 ** 6) == 1000
    assert P.snap(P.SETTING_SPECS["welt_select"], "gibt-es-nicht") is None and P.snap(P.SETTING_SPECS["welt_select"], "betrag") == "betrag"
    assert P.snap(P.SETTING_SPECS["seed_input"], -5) == 0 and P.snap(P.SETTING_SPECS["seed_input"], 10 ** 9) == C.SEED_MAX
