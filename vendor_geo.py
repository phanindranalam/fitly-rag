# VENDORED FROM THE FITLY PROJECT -- do not edit here.
#
# Copied rather than imported so this repo deploys standalone: Streamlit Cloud
# clones one repository, and an import reaching into ../fitly would work on a
# laptop and fail in production. Both modules are framework-independent (no
# Streamlit, no Supabase, no pandas), which is what makes vendoring clean.
#
# Source: https://github.com/phanindranalam/fitly
# To resync:  cp ../fitly/geo.py vendor_geo.py  then re-add this header.

"""Geography intent, job-location classification, and eligibility.

Framework-independent (no Streamlit/Supabase/pandas imports) so the whole
U.S.-first rule is unit-testable without booting the app -- see
tests/test_geo.py, which encodes the product acceptance criteria.

THE PRODUCT RULE THIS MODULE EXISTS TO ENFORCE
----------------------------------------------
Home country first, and "search there first" is not the same as "show those
jobs first". A user in Atlanta who picks Remote means "Remote -- United
States", not "Remote -- anywhere on earth"; a user in Bangalore who picks
Remote means "Remote -- India". The U.S. is the DEFAULT because it is the
most common answer, never because it is the right answer for everybody --
see SearchIntent.home_country, and the tests covering a non-U.S. user. Geography is therefore decided BEFORE fitment scoring,
not as a tiebreaker after it: a 98%-fit Toronto role is not a better result
than a 70%-fit Atlanta role for someone who never asked to look at Canada,
it is the wrong result.

Concretely that means the pipeline is:

    retrieve -> classify geography -> gate -> dedupe -> score fit -> rank

and not:

    retrieve -> score fit -> rank -> sort US first

WHAT "UNKNOWN" MEANS AND WHY IT IS NOT "US"
-------------------------------------------
A posting that says only "Remote" has NOT told us it is U.S.-eligible.
Treating that as U.S. would quietly recreate the bug this module prevents,
just with extra steps. So unknown geography is its own outcome
(Eligibility.UNKNOWN). It is shown, badged honestly as unclear, and ranked
below anything with confirmed eligibility -- hiding it outright would throw
away real roles, since a large share of legitimate U.S. postings are lazily
labelled "Remote" and nothing more. Callers that want the strict reading
can pass strict_unknown=True to drop them entirely.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not geocode, so radius filtering is textual (city/state token
match), not distance in miles. LocationIntent carries radius_miles because
the product spec calls for it and a future geocoding pass would use it, but
nothing here computes distance -- see _city_state_match. Claiming
"12 miles away" in the UI would be inventing a number, so the UI does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska",
    "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}
_STATE_NAME_TO_CODE = {name: code for code, name in US_STATES.items()}

# Ambiguous two-letter state codes that are also country codes or common
# words. Only trusted as a U.S. state when the string also carries another
# U.S. signal (see _looks_us). "IN" is Indiana and India's ISO code; "DE" is
# Delaware and Germany's; "CA" is California and Canada's; "OR"/"IT"/"ME"/
# "IN"/"OH"/"HI" are English words. Getting this wrong is exactly the class
# of bug that puts a Bangalore role in an Atlanta search.
_AMBIGUOUS_STATE_CODES = {"in", "de", "ca", "or", "me", "oh", "hi", "la", "ok", "id", "ma", "pa", "va", "wa", "mo", "mt", "co", "ar", "al", "ne", "sc", "nc"}

US_ALIASES = {
    "us", "usa", "u.s.", "u.s.a.", "united states", "united states of america",
    "america", "us only", "usa only", "us-based", "usa-based", "us based",
    "anywhere in the us", "anywhere in the usa", "remote us", "us remote",
}

# Country aliases -> ISO-3166 alpha-2. Deliberately covers the countries the
# app's live sources actually surface (Arbeitnow skews German/EU; RemoteOK,
# Jobicy and The Muse are global) plus the Adzuna market list, rather than
# every country on earth -- an unrecognized country name falls through to
# UNKNOWN, which is the safe outcome, not a silent U.S. match.
COUNTRY_ALIASES = {
    "canada": "ca", "ca (canada)": "ca", "toronto": "ca", "vancouver": "ca",
    "montreal": "ca", "ottawa": "ca", "calgary": "ca",
    "united kingdom": "gb", "uk": "gb", "u.k.": "gb", "great britain": "gb",
    "england": "gb", "scotland": "gb", "wales": "gb", "london": "gb",
    "northern ireland": "gb", "manchester": "gb", "edinburgh": "gb",
    "ireland": "ie", "dublin": "ie",
    "germany": "de", "deutschland": "de", "berlin": "de", "munich": "de",
    "münchen": "de", "hamburg": "de", "frankfurt": "de", "cologne": "de",
    "france": "fr", "paris": "fr", "lyon": "fr",
    "netherlands": "nl", "the netherlands": "nl", "amsterdam": "nl",
    "holland": "nl", "rotterdam": "nl",
    "austria": "at", "vienna": "at", "wien": "at",
    "belgium": "be", "brussels": "be",
    "spain": "es", "madrid": "es", "barcelona": "es",
    "portugal": "pt", "lisbon": "pt", "porto": "pt",
    "italy": "it", "rome": "it", "milan": "it",
    "switzerland": "ch", "zurich": "ch", "zürich": "ch", "geneva": "ch",
    "poland": "pl", "warsaw": "pl", "krakow": "pl", "kraków": "pl",
    "sweden": "se", "stockholm": "se",
    "norway": "no", "oslo": "no",
    "denmark": "dk", "copenhagen": "dk",
    "finland": "fi", "helsinki": "fi",
    "czech republic": "cz", "czechia": "cz", "prague": "cz",
    "romania": "ro", "bucharest": "ro",
    "ukraine": "ua", "kyiv": "ua", "kiev": "ua",
    "india": "in", "bangalore": "in", "bengaluru": "in", "hyderabad": "in",
    "mumbai": "in", "delhi": "in", "new delhi": "in", "pune": "in",
    "chennai": "in", "gurgaon": "in", "gurugram": "in", "noida": "in",
    "australia": "au", "sydney": "au", "melbourne": "au", "brisbane": "au",
    "new zealand": "nz", "auckland": "nz", "wellington": "nz",
    "singapore": "sg",
    "japan": "jp", "tokyo": "jp",
    "china": "cn", "beijing": "cn", "shanghai": "cn",
    "hong kong": "hk",
    "south korea": "kr", "seoul": "kr",
    "philippines": "ph", "manila": "ph",
    "indonesia": "id", "jakarta": "id",
    "vietnam": "vn", "hanoi": "vn",
    "malaysia": "my", "kuala lumpur": "my",
    "thailand": "th", "bangkok": "th",
    "israel": "il", "tel aviv": "il",
    "united arab emirates": "ae", "uae": "ae", "dubai": "ae", "abu dhabi": "ae",
    "south africa": "za", "cape town": "za", "johannesburg": "za",
    "nigeria": "ng", "lagos": "ng",
    "kenya": "ke", "nairobi": "ke",
    "egypt": "eg", "cairo": "eg",
    "brazil": "br", "brasil": "br", "sao paulo": "br", "são paulo": "br",
    "mexico": "mx", "mexico city": "mx", "ciudad de méxico": "mx",
    "argentina": "ar", "buenos aires": "ar",
    "chile": "cl", "santiago": "cl",
    "colombia": "co", "bogota": "co", "bogotá": "co",
    "peru": "pe", "lima": "pe",
    "turkey": "tr", "istanbul": "tr",
    "russia": "ru", "moscow": "ru",
    "pakistan": "pk", "karachi": "pk", "lahore": "pk",
    "bangladesh": "bd", "dhaka": "bd",
}

COUNTRY_NAMES = {
    "us": "United States", "ca": "Canada", "gb": "United Kingdom",
    "ie": "Ireland", "de": "Germany", "fr": "France", "nl": "Netherlands",
    "at": "Austria", "be": "Belgium", "es": "Spain", "pt": "Portugal",
    "it": "Italy", "ch": "Switzerland", "pl": "Poland", "se": "Sweden",
    "no": "Norway", "dk": "Denmark", "fi": "Finland", "cz": "Czech Republic",
    "ro": "Romania", "ua": "Ukraine", "in": "India", "au": "Australia",
    "nz": "New Zealand", "sg": "Singapore", "jp": "Japan", "cn": "China",
    "hk": "Hong Kong", "kr": "South Korea", "ph": "Philippines",
    "id": "Indonesia", "vn": "Vietnam", "my": "Malaysia", "th": "Thailand",
    "il": "Israel", "ae": "United Arab Emirates", "za": "South Africa",
    "ng": "Nigeria", "ke": "Kenya", "eg": "Egypt", "br": "Brazil",
    "mx": "Mexico", "ar": "Argentina", "cl": "Chile", "co": "Colombia",
    "pe": "Peru", "tr": "Turkey", "ru": "Russia", "pk": "Pakistan",
    "bd": "Bangladesh",
}

# Multi-country regions. Critically, "North America" is NOT a synonym for
# "United States" -- a North-America-wide posting may be Canada- or
# Mexico-payrolled, so it resolves to REGION and needs its own eligibility
# call, never an automatic U.S. pass.
REGIONS = {
    "north america": {"us", "ca", "mx"},
    "americas": {"us", "ca", "mx", "br", "ar", "cl", "co", "pe"},
    "latin america": {"mx", "br", "ar", "cl", "co", "pe"},
    "latam": {"mx", "br", "ar", "cl", "co", "pe"},
    "south america": {"br", "ar", "cl", "co", "pe"},
    "emea": {"gb", "ie", "de", "fr", "nl", "at", "be", "es", "pt", "it", "ch", "pl", "se", "no", "dk", "fi", "cz", "ro", "ua", "il", "ae", "za", "ng", "ke", "eg"},
    "europe": {"gb", "ie", "de", "fr", "nl", "at", "be", "es", "pt", "it", "ch", "pl", "se", "no", "dk", "fi", "cz", "ro", "ua"},
    "eu": {"ie", "de", "fr", "nl", "at", "be", "es", "pt", "it", "pl", "se", "dk", "fi", "cz", "ro"},
    "european union": {"ie", "de", "fr", "nl", "at", "be", "es", "pt", "it", "pl", "se", "dk", "fi", "cz", "ro"},
    "apac": {"au", "nz", "sg", "jp", "cn", "hk", "kr", "ph", "id", "vn", "my", "th", "in"},
    "asia": {"sg", "jp", "cn", "hk", "kr", "ph", "id", "vn", "my", "th", "in", "pk", "bd"},
    "asia pacific": {"au", "nz", "sg", "jp", "cn", "hk", "kr", "ph", "id", "vn", "my", "th", "in"},
    "africa": {"za", "ng", "ke", "eg"},
    "middle east": {"il", "ae", "tr"},
}

GLOBAL_PHRASES = {
    "anywhere", "worldwide", "world wide", "global", "globally",
    "work from anywhere", "anywhere in the world", "remote worldwide",
    "remote global", "remote anywhere", "any location", "location independent",
    "fully remote worldwide", "international",
}

REMOTE_PHRASES = {"remote", "fully remote", "100% remote", "work from home", "wfh", "telecommute", "distributed"}
HYBRID_PHRASES = {"hybrid", "partially remote", "flexible"}

# Work-authorization signals. Captured and surfaced separately -- NEVER
# folded into the eligibility decision. "Sponsorship not available" does not
# make a job U.S.-ineligible, it makes it a job whose sponsorship terms the
# user should see before spending an evening on the application.
_WORK_AUTH_PATTERNS = [
    (r"\bno (?:visa )?sponsorship\b|\bsponsorship (?:is )?not (?:available|provided|offered)\b|\bunable to sponsor\b|\bdo(?:es)? not (?:provide|offer) sponsorship\b", "no_sponsorship", "Sponsorship not provided"),
    (r"\b(?:visa )?sponsorship (?:is )?available\b|\bwill(?:ing to)? sponsor\b|\bwe sponsor\b", "sponsorship_available", "Sponsorship available"),
    (r"\bmust be (?:legally )?authorized to work in the (?:u\.?s\.?|united states)\b|\b(?:u\.?s\.?|united states) work authorization required\b|\bauthorization to work in the (?:u\.?s\.?|united states)\b", "us_auth_required", "U.S. work authorization required"),
    (r"\b(?:u\.?s\.?|united states) citizen(?:ship)?(?: is)?(?: required)?\b|\bmust be a (?:u\.?s\.?|united states) citizen\b|\bu\.?s\.? person\b", "us_citizenship_required", "U.S. citizenship required"),
    (r"\bsecurity clearance\b|\b(?:ts/sci|top secret|secret clearance)\b|\bactive clearance\b", "clearance_required", "Security clearance required"),
]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class RemoteScope:
    """How wide a remote posting's geography actually is."""
    US = "US"                 # Remote -- United States
    US_STATES = "US_STATES"   # Remote -- specific U.S. states only
    COUNTRY = "COUNTRY"       # Remote -- one named non-U.S. country
    REGION = "REGION"         # Remote -- a multi-country region (NOT == U.S.)
    GLOBAL = "GLOBAL"         # Remote -- worldwide / work from anywhere
    UNKNOWN = "UNKNOWN"       # Says "Remote" and nothing else
    NOT_REMOTE = "NOT_REMOTE"


class WorkMode:
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Eligibility:
    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    UNKNOWN = "unknown"


@dataclass
class JobLocation:
    """Normalized geography for one posting."""
    raw_location: str = ""
    country: str | None = None          # ISO alpha-2, lowercase
    state: str | None = None            # 2-letter US code, lowercase
    city: str | None = None
    work_mode: str = WorkMode.UNKNOWN
    remote_scope: str = RemoteScope.NOT_REMOTE
    remote_states: frozenset[str] = frozenset()
    region_countries: frozenset[str] = frozenset()
    location_confidence: str = "low"    # high | medium | low
    location_source: str = "job_posting"

    @property
    def country_name(self) -> str | None:
        return COUNTRY_NAMES.get(self.country) if self.country else None

    def display(self) -> str:
        """Short honest label for the result card. Never upgrades a bare
        'Remote' into 'Remote -- United States'."""
        if self.remote_scope == RemoteScope.US:
            return "Remote — United States"
        if self.remote_scope == RemoteScope.US_STATES:
            states = ", ".join(sorted(s.upper() for s in self.remote_states))
            return f"Remote — {states}"
        if self.remote_scope == RemoteScope.GLOBAL:
            return "Remote — Worldwide"
        if self.remote_scope == RemoteScope.REGION:
            return f"Remote — {self.raw_location.strip()}"
        if self.remote_scope == RemoteScope.COUNTRY:
            return f"Remote — {self.country_name or self.raw_location.strip()}"
        if self.remote_scope == RemoteScope.UNKNOWN:
            return "Remote — geography unclear"
        return self.raw_location.strip() or "Location not stated"


@dataclass
class LocationIntent:
    """Where the USER wants to work. Defaults are deliberately U.S.-only."""
    country: str = "us"
    state: str | None = None
    city: str | None = None
    radius_miles: int = 50
    work_modes: tuple[str, ...] = (WorkMode.REMOTE, WorkMode.HYBRID, WorkMode.ONSITE)
    preferred_locations: tuple[str, ...] = ()

    @property
    def wants_remote(self) -> bool:
        return WorkMode.REMOTE in self.work_modes

    @property
    def wants_onsite_ish(self) -> bool:
        return WorkMode.HYBRID in self.work_modes or WorkMode.ONSITE in self.work_modes


@dataclass
class SearchIntent:
    """One search, fully specified. Every source adapter receives THIS, not
    a raw 'Atlanta, GA' string each adapter is free to reinterpret."""
    role: str = ""
    # Additional role titles to search alongside `role`, normally derived
    # from the user's resume. Senior titles are wildly inconsistent
    # ("Director of Platform Engineering", "Head of Infrastructure",
    # "Principal SRE") and every source here matches on the posting TITLE,
    # so a single string silently misses most of the market. A posting
    # matches if ANY variant matches, which raises recall without loosening
    # what counts as a match for any one variant.
    role_variants: tuple[str, ...] = ()
    location: LocationIntent = field(default_factory=LocationIntent)
    allowed_countries: frozenset[str] = frozenset({"us"})
    global_search: bool = False
    strict_unknown: bool = False

    @classmethod
    def us_only(cls, role: str = "", **loc_kwargs) -> "SearchIntent":
        return cls(role=role, location=LocationIntent(**loc_kwargs))

    @property
    def queries(self) -> tuple[str, ...]:
        """Every role title this search should look for, deduplicated and
        order-preserving with the typed role first."""
        seen, out = set(), []
        for q in (self.role, *self.role_variants):
            key = " ".join((q or "").lower().split())
            if key and key not in seen:
                seen.add(key)
                out.append(q.strip())
        return tuple(out)

    @property
    def home_country(self) -> str:
        """The user's own country. Everything geography-related is relative to
        this, not to the U.S.: "home country first" is the rule, and the U.S.
        is merely the most common answer to it."""
        return (self.location.country or "us").strip().lower()

    @property
    def international(self) -> bool:
        """True once the user has opted beyond their OWN country.

        Measured against home_country, not against the U.S. Comparing to
        {"us"} meant a user in India searching only India registered as an
        international search, which unlocked the global source tiers they
        had specifically not asked for.
        """
        return self.global_search or self.allowed_countries != frozenset({self.home_country})

    @property
    def geography_label(self) -> str:
        if self.global_search:
            return "Worldwide"
        home = self.home_country
        if self.allowed_countries == frozenset({home}):
            return f"{COUNTRY_NAMES.get(home, home.upper())} only"
        names = sorted(COUNTRY_NAMES.get(c, c.upper()) for c in self.allowed_countries)
        return " + ".join(names)


@dataclass
class EligibilityResult:
    status: str
    reason: str
    tier: int          # 1 = exact local match ... 6 = worldwide; 99 = excluded
    location_score: int  # 0-10, its own scoring dimension alongside fit


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[,/|;()\[\]\-–—]+|\s{2,}", s) if t.strip()]


# "america" only counts as a U.S. signal when it is not part of a larger
# region name. Without this, "North America" matched the U.S. alias list and
# a Canada-or-Mexico-payrolled posting sailed through a U.S.-only search --
# the precise mistake the spec calls out ("do NOT treat North America as
# equivalent to the United States").
_AMERICA_RE = re.compile(r"(?<!north )(?<!south )(?<!latin )(?<!central )\bamerica\b")
_US_TOKEN_RE = re.compile(r"\b(?:united states(?: of america)?|u\.?s\.?a\.?|us|usa)\b")


def _looks_us(text: str) -> bool:
    """Any independent signal that a string is about the United States,
    used to disambiguate two-letter codes that are both a U.S. state and a
    country (IN, DE, CA...)."""
    if text.strip() in US_ALIASES:
        return True
    if _US_TOKEN_RE.search(text):
        return True
    return bool(_AMERICA_RE.search(text))


def _strip_remote_words(text: str) -> str:
    """Reduce 'Remote — United States' to 'united states' so the alias
    tables can be matched exactly rather than by fuzzy substring."""
    cleaned = text
    for phrase in sorted(REMOTE_PHRASES | HYBRID_PHRASES, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned)
    cleaned = re.sub(r"^[\s,\-–—:()/|]+|[\s,\-–—:()/|]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_work_mode(raw_location: str, *extra_text: str) -> str:
    hay = " ".join(_norm(t) for t in (raw_location, *extra_text) if t)
    loc_only = _norm(raw_location)
    if any(p in hay for p in HYBRID_PHRASES):
        return WorkMode.HYBRID
    if any(re.search(rf"\b{re.escape(p)}\b", hay) for p in REMOTE_PHRASES):
        return WorkMode.REMOTE
    # "Worldwide", "Anywhere in the US", "Work from anywhere" are remote
    # postings that never use the word "remote". Checked against the
    # location field only -- a description mentioning "anywhere" in passing
    # is not a work-mode declaration.
    if any(re.search(rf"(?:^|[^a-z]){re.escape(p)}(?:[^a-z]|$)", loc_only) for p in GLOBAL_PHRASES):
        return WorkMode.REMOTE
    if raw_location.strip():
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN


def extract_work_authorization(*texts: str) -> list[tuple[str, str]]:
    """Returns [(code, human_label)] for every authorization signal found.
    Never consulted by evaluate_eligibility -- surfaced to the user instead
    so they decide, which is the whole point of not silently inferring."""
    hay = " ".join(_norm(t) for t in texts if t)
    found = []
    for pattern, code, label in _WORK_AUTH_PATTERNS:
        if re.search(pattern, hay):
            found.append((code, label))
    return found


def _split_tokens(text: str) -> list[str]:
    # Split on dashes as well as commas: real postings write "Remote — CA,
    # TX, NY", so a comma-only split leaves the first state welded to the
    # word "Remote" ("remote — ca") and silently drops it.
    return [t.strip().strip(".") for t in re.split(r"[,/|;\-–—()\[\]]+|\s+or\s+|\s+and\s+", text) if t.strip()]


def _detect_foreign_country(text: str) -> tuple[str | None, str, str | None]:
    """Look for an explicit non-U.S. country or a city that identifies one.

    Runs BEFORE U.S. state parsing so "Toronto, CA" resolves via Toronto
    (Canada) rather than via CA (California). Returns
    (country_code, confidence, city_token).
    """
    for tok in sorted(_split_tokens(text), key=len, reverse=True):
        if tok in COUNTRY_ALIASES:
            code = COUNTRY_ALIASES[tok]
            city = tok if len(tok) > 2 and tok not in {n.lower() for n in COUNTRY_NAMES.values()} else None
            return code, "high", city
    for alias, code in COUNTRY_ALIASES.items():
        if len(alias) > 3 and re.search(rf"(?:^|[^a-z]){re.escape(alias)}(?:[^a-z]|$)", text):
            return code, "medium", None
    return None, "low", None


def _parse_us_states(text: str, *, allow_ambiguous: bool = False) -> frozenset[str]:
    """Pull explicit state codes/names out of e.g. 'Remote - CA, TX, NY'.

    Two passes, because half the state codes are also country codes or
    English words (CA/Canada, IN/India, DE/Germany, OR, ME...). An
    ambiguous code is only accepted when the string has proven itself
    U.S.-flavored some other way:

      - an unambiguous state sits alongside it ("CA, TX, NY" -- TX and NY
        settle it),
      - an explicit U.S. token is present ("Sacramento, CA, USA"), or
      - allow_ambiguous, which the caller sets only after confirming no
        non-U.S. country matched. That is what lets "San Francisco, CA"
        resolve as California while "Toronto, CA" stays Canada: Toronto is
        a known Canadian city, San Francisco matches no foreign country.

    The residual risk is an unlisted foreign city paired with a country
    code that collides with a U.S. state ("Ulm, DE"). parse_location marks
    that path medium-confidence rather than high so the UI can badge it,
    since the failure direction -- a German role reading as U.S. -- is the
    expensive one.
    """
    tokens = _split_tokens(text)

    unambiguous = set()
    ambiguous = set()
    for t in tokens:
        if t in _STATE_NAME_TO_CODE:
            unambiguous.add(_STATE_NAME_TO_CODE[t])
        elif len(t) == 2 and t in US_STATES:
            (ambiguous if t in _AMBIGUOUS_STATE_CODES else unambiguous).add(t)

    if unambiguous or allow_ambiguous or _looks_us(text):
        return frozenset(unambiguous | ambiguous)
    return frozenset(unambiguous)


def parse_location(raw_location: str, *, description: str = "", source_is_remote_only: bool = False) -> JobLocation:
    """Classify one posting's raw location string.

    source_is_remote_only marks feeds that only ever carry remote roles
    (RemoteOK, Jobicy). It settles work_mode, and nothing else -- it must
    never be read as evidence of WHICH geography, which is precisely the
    conflation ('remote board, so it's fine for me') this module blocks.
    """
    raw = (raw_location or "").strip()
    text = _norm(raw)
    loc = JobLocation(raw_location=raw)

    loc.work_mode = detect_work_mode(raw, description if len(description) < 4000 else description[:4000])
    if source_is_remote_only and loc.work_mode in (WorkMode.UNKNOWN, WorkMode.ONSITE):
        loc.work_mode = WorkMode.REMOTE

    if not text:
        loc.remote_scope = RemoteScope.UNKNOWN if loc.work_mode == WorkMode.REMOTE else RemoteScope.NOT_REMOTE
        loc.location_confidence = "low"
        return loc

    is_remote = loc.work_mode == WorkMode.REMOTE
    bare = _strip_remote_words(text)

    # 1. An exact U.S. alias, checked before the worldwide phrases so
    #    "Anywhere in the US" reads as U.S.-wide rather than tripping the
    #    "anywhere" keyword and being classified as worldwide.
    if bare in US_ALIASES or text in US_ALIASES:
        loc.country = "us"
        loc.location_confidence = "high"
        loc.remote_scope = RemoteScope.US if is_remote else RemoteScope.NOT_REMOTE
        return loc

    # 2. Worldwide -- checked early so "Remote (Anywhere)" never falls
    #    through to a city lookup and gets mistaken for somewhere specific.
    if any(re.search(rf"(?:^|[^a-z]){re.escape(p)}(?:[^a-z]|$)", text) for p in GLOBAL_PHRASES):
        loc.remote_scope = RemoteScope.GLOBAL if is_remote else RemoteScope.NOT_REMOTE
        loc.location_confidence = "high"
        return loc

    # 3. A named region. Deliberately BEFORE both the U.S. check and the
    #    country lookup: "North America" contains the token "America" and
    #    would otherwise match the U.S. alias list, quietly passing a
    #    Canada- or Mexico-payrolled posting through a U.S.-only search.
    for region, members in REGIONS.items():
        if re.search(rf"(?:^|[^a-z]){re.escape(region)}(?:[^a-z]|$)", text):
            loc.region_countries = frozenset(members)
            loc.remote_scope = RemoteScope.REGION if is_remote else RemoteScope.NOT_REMOTE
            loc.location_confidence = "medium"
            return loc

    # 4. An explicit non-U.S. country, or a city that identifies one.
    #    Ahead of state parsing so "Toronto, CA" is Canada, not California.
    foreign, foreign_confidence, foreign_city = _detect_foreign_country(text)
    if foreign and foreign != "us":
        loc.country = foreign
        loc.city = foreign_city
        loc.location_confidence = foreign_confidence
        loc.remote_scope = RemoteScope.COUNTRY if is_remote else RemoteScope.NOT_REMOTE
        return loc

    # 5. Explicit U.S. state list ("Remote - CA, TX, NY", "San Francisco, CA").
    #    Ambiguous codes are unlocked here only because step 4 already ruled
    #    out every foreign country this app knows about.
    states = _parse_us_states(text, allow_ambiguous=True)
    if states:
        strong_us = bool(_looks_us(text)) or bool(_parse_us_states(text))
        loc.country = "us"
        loc.remote_states = states
        # High only when something other than an ambiguous code carried the
        # decision; otherwise medium, so the UI can badge a "City, XX" guess
        # honestly instead of overstating it.
        loc.location_confidence = "high" if strong_us else "medium"
        if len(states) == 1:
            loc.state = next(iter(states))
        if is_remote:
            loc.remote_scope = RemoteScope.US_STATES if len(states) > 1 or not _looks_us(text) else RemoteScope.US
        else:
            loc.remote_scope = RemoteScope.NOT_REMOTE
        loc.city = _extract_city(text, states)
        return loc

    # 6. Whole-country U.S. ("Remote, United States", "USA").
    if _looks_us(text):
        loc.country = "us"
        loc.location_confidence = "high"
        loc.remote_scope = RemoteScope.US if is_remote else RemoteScope.NOT_REMOTE
        return loc

    # 7. Nothing identifiable. "Remote" with no geography lands here, and
    #    stays UNKNOWN rather than being charitably read as U.S.
    loc.remote_scope = RemoteScope.UNKNOWN if is_remote else RemoteScope.NOT_REMOTE
    loc.location_confidence = "low"
    if not is_remote:
        loc.city = text.split(",")[0].strip() or None
    return loc


def _extract_city(text: str, states: frozenset[str]) -> str | None:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for p in parts:
        if p in US_STATES or p in _STATE_NAME_TO_CODE or p in US_ALIASES:
            continue
        if any(re.search(rf"\b{re.escape(w)}\b", p) for w in REMOTE_PHRASES | HYBRID_PHRASES):
            continue
        return p
    return None


# ---------------------------------------------------------------------------
# Eligibility -- the gate
# ---------------------------------------------------------------------------

def _city_state_match(loc: JobLocation, intent: LocationIntent) -> bool:
    """Textual city/state match. NOT a radius calculation -- see the module
    docstring. A city the user named matches; anything else in the same
    state counts as a state-level match, handled by the caller."""
    want_city = _norm(intent.city)
    if not want_city:
        return False
    hay = " ".join(filter(None, [_norm(loc.city), _norm(loc.raw_location)]))
    return bool(hay) and want_city in hay


def evaluate_eligibility(loc: JobLocation, intent: SearchIntent) -> EligibilityResult:
    """Decide whether one posting may enter the result set at all, and at
    what priority tier. Called BEFORE fitment scoring.

    Tiers follow the product spec:
      1 exact local match, 2 preferred user location, 3 U.S. remote,
      4 other U.S. location, 5 explicitly selected international, 6 worldwide.
    """
    allowed = set(intent.allowed_countries)
    li = intent.location
    us_only = not intent.global_search and allowed == {"us"}

    # --- Work-mode gate: applies regardless of geography. ---
    if loc.work_mode == WorkMode.REMOTE and not li.wants_remote:
        return EligibilityResult(Eligibility.EXCLUDED, "Remote role, but Remote isn't a selected workplace type", 99, 0)
    if loc.work_mode in (WorkMode.ONSITE, WorkMode.HYBRID) and not li.wants_onsite_ish:
        return EligibilityResult(Eligibility.EXCLUDED, "On-site/hybrid role, but only Remote is selected", 99, 0)

    # --- Worldwide postings. ---
    if loc.remote_scope == RemoteScope.GLOBAL:
        if intent.global_search:
            return EligibilityResult(Eligibility.ELIGIBLE, "Worldwide remote, worldwide search enabled", 6, 6)
        # A genuinely worldwide role does include the U.S., so this is
        # eligible even under U.S.-only -- but it sits at the bottom tier,
        # because "open to everyone on earth" is weaker evidence of real
        # U.S. eligibility than "Remote -- United States".
        return EligibilityResult(Eligibility.ELIGIBLE, "Worldwide remote — includes the U.S., but eligibility varies by payroll entity", 6, 5)

    # --- Unknown geography: never assumed to be U.S. ---
    if loc.remote_scope == RemoteScope.UNKNOWN or (loc.country is None and not loc.region_countries):
        if intent.strict_unknown:
            return EligibilityResult(Eligibility.EXCLUDED, "Geography could not be established and strict mode is on", 99, 0)
        reason = (
            "Posting says Remote without stating a geography — U.S. eligibility not verified"
            if loc.remote_scope == RemoteScope.UNKNOWN
            else (
                f"Couldn't place \"{loc.raw_location.strip()}\" — eligibility not verified"
                if loc.raw_location.strip()
                else "Posting states no location — eligibility not verified"
            )
        )
        return EligibilityResult(Eligibility.UNKNOWN, reason, 6, 3)

    # --- Regions: a region is eligible only if it overlaps the allowed set,
    #     and even then it is not treated as a confirmed country match. ---
    if loc.region_countries:
        if intent.global_search:
            return EligibilityResult(Eligibility.ELIGIBLE, "Regional remote, worldwide search enabled", 6, 5)
        overlap = loc.region_countries & allowed
        if not overlap:
            return EligibilityResult(Eligibility.EXCLUDED, f"Region does not cover your selected geography ({intent.geography_label})", 99, 0)
        return EligibilityResult(Eligibility.UNKNOWN, "Regional posting — covers your geography, but per-country eligibility isn't stated", 5, 4)

    country = loc.country

    # --- Country gate. This is the line that keeps a 98%-fit Toronto role
    #     out of an Atlanta search. ---
    if not intent.global_search and country not in allowed:
        return EligibilityResult(
            Eligibility.EXCLUDED,
            f"{COUNTRY_NAMES.get(country, (country or '').upper())} is outside your selected geography ({intent.geography_label})",
            99, 0,
        )

    # --- Non-U.S. but explicitly allowed. ---
    if country != "us":
        return EligibilityResult(Eligibility.ELIGIBLE, f"{COUNTRY_NAMES.get(country, country.upper())} — a geography you selected", 5, 6)

    # --- U.S. jobs, tiered. ---
    if loc.remote_scope == RemoteScope.US_STATES:
        want_state = _norm(li.state)
        want_code = _STATE_NAME_TO_CODE.get(want_state, want_state)
        if want_code and want_code in loc.remote_states:
            return EligibilityResult(Eligibility.ELIGIBLE, f"Remote, open to {want_code.upper()}", 3, 10)
        if want_code:
            states = ", ".join(sorted(s.upper() for s in loc.remote_states))
            return EligibilityResult(Eligibility.EXCLUDED, f"Remote but restricted to {states}, which doesn't include {want_code.upper()}", 99, 0)
        return EligibilityResult(Eligibility.UNKNOWN, "Remote but state-restricted — add your state to check eligibility", 3, 5)

    if loc.remote_scope == RemoteScope.US:
        return EligibilityResult(Eligibility.ELIGIBLE, "Remote — United States", 3, 10)

    # On-site / hybrid inside the U.S.
    if _city_state_match(loc, li):
        return EligibilityResult(Eligibility.ELIGIBLE, f"In {li.city}", 1, 10)

    want_state = _norm(li.state)
    want_code = _STATE_NAME_TO_CODE.get(want_state, want_state)
    if want_code and loc.state == want_code:
        return EligibilityResult(Eligibility.ELIGIBLE, f"Elsewhere in {want_code.upper()}", 2, 8)

    for pref in li.preferred_locations:
        if _norm(pref) and _norm(pref) in _norm(loc.raw_location):
            return EligibilityResult(Eligibility.ELIGIBLE, f"In {pref}, one of your preferred locations", 2, 9)

    # A U.S. job somewhere the user didn't ask for. Eligible only when the
    # user gave no city at all; otherwise it is a location mismatch, and
    # the spec is explicit that those get filtered, not badged.
    if not li.city and not want_code:
        return EligibilityResult(Eligibility.ELIGIBLE, "United States", 4, 7)
    return EligibilityResult(
        Eligibility.EXCLUDED,
        f"{loc.raw_location.strip() or 'This U.S. location'} is outside {li.city or want_code.upper()}",
        99, 0,
    )


def classify(raw_location: str, intent: SearchIntent, *, description: str = "", source_is_remote_only: bool = False) -> tuple[JobLocation, EligibilityResult]:
    """Convenience: parse then gate, in the one call the retrieval layer needs."""
    loc = parse_location(raw_location, description=description, source_is_remote_only=source_is_remote_only)
    return loc, evaluate_eligibility(loc, intent)
