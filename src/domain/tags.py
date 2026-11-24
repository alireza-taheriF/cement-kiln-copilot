"""Canonical tag dictionary.

Every plant names its DCS/historian tags differently (``K1_BZT_PV``,
``431KI01.T``...). The copilot works on the *canonical* names below; the
connector layer (:mod:`src.connectors`) maps plant tags onto them, with unit
conversion, through ``config/connectors.yaml``.

Keeping one dictionary means models, safety limits, KPIs and the dashboard
all agree on what a tag means and which unit it is in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TagKind(str, Enum):
    """Role of a tag in the process."""

    PROCESS = "process"  # measured, not directly set by the operator
    SETPOINT = "setpoint"  # operator-adjustable, the copilot may advise on it
    DERIVED = "derived"  # computed from other tags (KPIs)
    LAB = "lab"  # laboratory result, irregular sampling


@dataclass(frozen=True)
class TagSpec:
    """Metadata of one canonical tag."""

    name: str
    unit: str
    kind: TagKind
    description: str
    description_fa: str


# --- Kiln line --------------------------------------------------------------
KILN_FEED_TPH = "KILN_FEED_TPH"
CLINKER_TPH = "CLINKER_TPH"
GAS_FLOW_NM3H = "GAS_FLOW_NM3H"
MAZUT_FLOW_KGH = "MAZUT_FLOW_KGH"
KILN_SPEED_RPM = "KILN_SPEED_RPM"
ID_FAN_PCT = "ID_FAN_PCT"
BZ_TEMP_C = "BZ_TEMP_C"
KILN_INLET_O2_PCT = "KILN_INLET_O2_PCT"
KILN_INLET_CO_PPM = "KILN_INLET_CO_PPM"
NOX_MGNM3 = "NOX_MGNM3"
SEC_AIR_TEMP_C = "SEC_AIR_TEMP_C"

# --- Lab --------------------------------------------------------------------
FREE_LIME_PCT = "FREE_LIME_PCT"

# --- Cement mills -----------------------------------------------------------
MILL_KW_TEMPLATE = "CEMENT_MILL{n}_KW"
MILL_TPH_TEMPLATE = "CEMENT_MILL{n}_TPH"
MILL_VIBRATION_TEMPLATE = "MILL{n}_VIBRATION_MMS"
MILL_BEARING_TEMPLATE = "MILL{n}_BEARING_TEMP_C"
CEMENT_SILO_T = "CEMENT_SILO_T"

# --- Kiln shell scanner -----------------------------------------------------
SHELL_TEMP_TEMPLATE = "SHELL_TEMP_P{pos:02d}"

# --- Derived KPIs (see src.domain.kpis) -------------------------------------
HEAT_INPUT_GCALH = "HEAT_INPUT_GCALH"
GAS_HEAT_SHARE = "GAS_HEAT_SHARE"
SPECIFIC_HEAT_KCAL_KG = "SPECIFIC_HEAT_KCAL_KG"
MILL_SPECIFIC_KWH_T_TEMPLATE = "CEMENT_MILL{n}_KWH_T"


def mill_tag(template: str, n: int) -> str:
    """Render a per-mill tag name, e.g. ``mill_tag(MILL_KW_TEMPLATE, 1)``."""
    return template.format(n=n)


def shell_tag(position: int) -> str:
    """Render the shell-scanner tag for an axial position (0 = kiln outlet)."""
    return SHELL_TEMP_TEMPLATE.format(pos=position)


TAGS: dict[str, TagSpec] = {
    spec.name: spec
    for spec in (
        TagSpec(KILN_FEED_TPH, "t/h", TagKind.PROCESS, "Kiln raw-meal feed", "خوراک کوره"),
        TagSpec(CLINKER_TPH, "t/h", TagKind.PROCESS, "Clinker production", "تولید کلینکر"),
        TagSpec(GAS_FLOW_NM3H, "Nm3/h", TagKind.SETPOINT, "Natural gas to main burner", "دبی گاز مشعل"),
        TagSpec(MAZUT_FLOW_KGH, "kg/h", TagKind.SETPOINT, "Heavy fuel oil to main burner", "دبی مازوت مشعل"),
        TagSpec(KILN_SPEED_RPM, "rpm", TagKind.SETPOINT, "Kiln rotation speed", "دور کوره"),
        TagSpec(ID_FAN_PCT, "%", TagKind.SETPOINT, "ID fan speed", "دور فن ID"),
        TagSpec(BZ_TEMP_C, "°C", TagKind.PROCESS, "Burning-zone temperature", "دمای زون پخت"),
        TagSpec(KILN_INLET_O2_PCT, "%", TagKind.PROCESS, "Kiln inlet O2", "اکسیژن ورودی کوره"),
        TagSpec(KILN_INLET_CO_PPM, "ppm", TagKind.PROCESS, "Kiln inlet CO", "CO ورودی کوره"),
        TagSpec(NOX_MGNM3, "mg/Nm3", TagKind.PROCESS, "Stack NOx", "NOx دودکش"),
        TagSpec(SEC_AIR_TEMP_C, "°C", TagKind.PROCESS, "Secondary air temperature", "دمای هوای ثانویه"),
        TagSpec(FREE_LIME_PCT, "%", TagKind.LAB, "Clinker free lime (lab)", "آهک آزاد کلینکر"),
        TagSpec(CEMENT_SILO_T, "t", TagKind.PROCESS, "Cement silo inventory", "موجودی سیلو سیمان"),
        TagSpec(HEAT_INPUT_GCALH, "Gcal/h", TagKind.DERIVED, "Total kiln heat input", "حرارت ورودی کل"),
        TagSpec(GAS_HEAT_SHARE, "-", TagKind.DERIVED, "Gas share of heat input", "سهم گاز از حرارت"),
        TagSpec(
            SPECIFIC_HEAT_KCAL_KG,
            "kcal/kg",
            TagKind.DERIVED,
            "Specific heat consumption per kg clinker",
            "مصرف حرارت ویژه",
        ),
    )
}


def describe(tag: str, lang: str = "en") -> str:
    """Human label for a tag, falling back to the raw name for unknown tags."""
    spec = TAGS.get(tag)
    if spec is None:
        return tag
    return spec.description_fa if lang == "fa" else spec.description
