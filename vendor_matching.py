# VENDORED FROM THE FITLY PROJECT -- do not edit here.
#
# Copied rather than imported so this repo deploys standalone: Streamlit Cloud
# clones one repository, and an import reaching into ../fitly would work on a
# laptop and fail in production. Both modules are framework-independent (no
# Streamlit, no Supabase, no pandas), which is what makes vendoring clean.
#
# Source: https://github.com/phanindranalam/fitly
# To resync:  cp ../fitly/matching.py vendor_matching.py  then re-add this header.

"""Free, explainable keyword-based fitment scoring. No LLM calls, no cost,
no external API -- framework-independent (no Streamlit/Supabase imports).

Deliberately a curated skills taxonomy rather than raw word-frequency/TF-IDF:
a frequency-based approach produces noisy "gaps" like "team" or "experience".
Matching against a known vocabulary keeps the output explainable: every gap
shown is a real, recognizable skill term, not a stray common word.
"""

import re
from dataclasses import dataclass, field

_LANGUAGES = [
    "python", "java", "javascript", "typescript", "sql", "c++", "c#", "go",
    "golang", "rust", "ruby", "php", "scala", "r", "swift", "kotlin", "bash",
    "shell", "powershell", "perl", "elixir", "dart", "objective-c",
]
_DATA_ML = [
    "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "keras",
    "machine learning", "deep learning", "nlp", "llm", "large language model",
    "data analysis", "data science", "data engineering", "etl", "spark",
    "hadoop", "airflow", "tableau", "power bi", "looker", "excel",
    "statistics", "a/b testing", "data visualization", "dbt", "snowflake",
    "bigquery", "redshift",
    "databricks", "delta lake", "iceberg", "clickhouse", "duckdb", "presto",
    "trino", "flink", "kinesis", "ray", "mlflow", "kubeflow", "sagemaker",
    "vertex ai", "bedrock", "langchain", "vector database", "embeddings",
    "rag", "fine-tuning", "prompt engineering", "feature store",
    "data warehouse", "data lake", "data modeling", "data governance",
    "data quality", "dagster", "prefect", "great expectations",
]
_CLOUD_PLATFORM = [
    "aws", "azure", "gcp", "google cloud", "kubernetes", "docker",
    "terraform", "ansible", "jenkins", "ci/cd", "gitops", "argocd",
    "cloudformation", "serverless", "lambda", "microservices", "linux",
    "networking", "devops", "sre", "site reliability", "observability",
    "prometheus", "grafana", "datadog",
    # Added after a live rollup showed Kafka and Go appearing in real
    # postings and scoring as neither a match nor a gap, because the
    # taxonomy simply didn't know the words. A term the scorer can't see is
    # invisible in both directions, which is worse than scoring it wrong.
    "kafka", "rabbitmq", "grpc", "istio", "helm", "vault", "consul",
    "nomad", "opentelemetry", "jaeger", "splunk", "elk", "loki",
    "pagerduty", "eks", "aks", "gke", "openshift", "rancher", "karpenter",
    "packer", "chef", "puppet", "spinnaker", "flux", "crossplane",
    "cilium", "envoy", "nginx", "load balancing", "service mesh",
    "infrastructure as code", "platform engineering", "capacity planning",
    "incident response", "on-call", "postmortem", "slo", "sli", "error budget",
    "disaster recovery", "high availability", "cost optimization", "finops",
    "github actions", "circleci", "argo workflows", "tekton", "bazel",
]
_WEB_APP = [
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "rest api", "graphql", "html", "css", "streamlit", "next.js",
]
_DATABASES = [
    "postgres", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    "supabase", "firebase", "elasticsearch", "database design", "nosql",
]
_METHODOLOGY = [
    "agile", "scrum", "kanban", "jira", "unit testing", "test automation",
    "tdd", "code review", "git", "github", "gitlab", "version control",
]
_BUSINESS_SOFT = [
    "project management", "product management", "stakeholder management",
    "communication", "leadership", "cross-functional", "roadmap",
    "requirements gathering", "presentation", "negotiation", "mentoring",
    "budget management", "vendor management", "customer success",
    "sales", "marketing", "seo", "content strategy", "ux research",
    "ui design", "figma", "wireframing", "user research",
]
_SECURITY_COMPLIANCE = [
    "security", "compliance", "soc 2", "gdpr", "hipaa", "penetration testing",
    "risk management", "identity and access management", "iam",
]

# The taxonomy above is tech-only, which quietly made this scorer useless for
# most of the workforce. Measured before adding the groups below: a
# registered nurse's resume against a real ICU posting extracted FOUR terms,
# three of them from the acronym regex rather than the taxonomy, and returned
# "100% fit" -- a number that only means "it matched everything it could see,
# and it could barely see anything". A marketer scored the same way on three
# terms. The confidence flag caught it, but reporting low confidence is a
# workaround for not knowing the vocabulary, not a fix.
#
# These groups are curated the same way as the tech ones: recognizable,
# checkable terms a human would agree are skills, so every match and gap
# stays explainable. They are not exhaustive and don't need to be -- the goal
# is enough vocabulary that the score reflects the role.
_HEALTHCARE = [
    "patient care", "patient assessment", "triage", "clinical", "icu",
    "emergency", "acute care", "med surg", "medication administration",
    "wound care", "care plan", "care coordination", "discharge planning",
    "phlebotomy", "vital signs", "infection control", "hipaa compliance",
    "electronic health record", "ehr", "emr", "epic", "cerner", "meditech",
    "acls", "bls", "pals", "cpr", "rn", "lpn", "bsn", "np", "pa-c",
    "charting", "telemetry", "iv therapy", "dosage calculation",
    "preceptor", "case management", "utilization review", "revenue cycle",
    "medical coding", "icd-10", "cpt", "prior authorization", "pharmacy",
    "radiology", "phlebotomist", "surgical", "perioperative", "anesthesia",
    "physical therapy", "occupational therapy", "behavioral health",
    "public health", "epidemiology", "clinical trials", "gcp", "irb",
]
_LEGAL = [
    "contract review", "contract negotiation", "litigation", "discovery",
    "deposition", "legal research", "westlaw", "lexisnexis", "due diligence",
    "regulatory compliance", "corporate governance", "intellectual property",
    "trademark", "patent", "licensing", "employment law", "privacy law",
    "ccpa", "sox", "anti-money laundering", "aml", "kyc", "e-discovery",
    "paralegal", "legal writing", "brief writing", "mediation", "arbitration",
]
_FINANCE_ACCOUNTING = [
    "financial modeling", "forecasting", "budgeting", "variance analysis",
    "financial reporting", "gaap", "ifrs", "month end close", "reconciliation",
    "accounts payable", "accounts receivable", "general ledger", "payroll",
    "audit", "internal controls", "tax preparation", "cost accounting",
    "fp&a", "treasury", "cash flow", "valuation", "dcf", "underwriting",
    "credit analysis", "portfolio management", "quickbooks", "netsuite",
    "sap fico", "oracle financials", "cpa", "cfa", "bloomberg terminal",
    "invoicing", "procurement", "capital planning", "revenue recognition",
]
_SALES_CS = [
    "prospecting", "lead generation", "cold calling", "pipeline management",
    "quota", "account management", "territory management", "upselling",
    "cross-selling", "contract negotiation", "salesforce", "hubspot crm",
    "outreach", "salesloft", "gong", "discovery calls", "demos",
    "solution selling", "consultative selling", "meddic", "spin selling",
    "channel partnerships", "renewals", "churn reduction", "onboarding",
    "customer retention", "escalation management", "zendesk", "intercom",
    "service level agreement", "nps", "voice of customer",
]
_MARKETING = [
    "content marketing", "content strategy", "copywriting", "brand strategy",
    "brand positioning", "campaign management", "email marketing",
    "marketing automation", "hubspot", "marketo", "mailchimp", "klaviyo",
    "paid media", "paid search", "sem", "ppc", "google ads", "meta ads",
    "programmatic", "social media", "influencer marketing", "seo",
    "keyword research", "google analytics", "ga4", "attribution",
    "conversion rate optimization", "landing pages", "a/b testing",
    "market research", "competitive analysis", "positioning", "messaging",
    "product marketing", "demand generation", "lifecycle marketing",
    "public relations", "event marketing", "webinars", "editorial calendar",
]
_OPERATIONS = [
    "supply chain", "logistics", "inventory management", "warehouse",
    "fulfillment", "procurement", "vendor management", "sourcing",
    "demand planning", "capacity planning", "lean", "six sigma", "kaizen",
    "process improvement", "continuous improvement", "root cause analysis",
    "standard operating procedure", "quality assurance", "quality control",
    "iso 9001", "erp", "sap", "oracle scm", "netsuite erp", "wms", "tms",
    "freight", "customs", "distribution", "forecast accuracy", "cogs",
    "manufacturing", "production planning", "maintenance", "safety",
    "osha", "facilities management", "fleet management",
]
_HR_PEOPLE = [
    "recruiting", "talent acquisition", "sourcing candidates", "screening",
    "interviewing", "onboarding", "employee relations", "performance management",
    "compensation", "benefits administration", "hris", "workday hcm",
    "successfactors", "greenhouse", "lever", "applicant tracking system",
    "employee engagement", "retention", "succession planning",
    "learning and development", "training delivery", "instructional design",
    "diversity and inclusion", "hr policy", "labor relations", "fmla",
    "workers compensation", "employment compliance", "organizational design",
    "change management", "headcount planning", "workforce planning",
]
_EDUCATION = [
    "curriculum development", "lesson planning", "classroom management",
    "differentiated instruction", "assessment design", "student engagement",
    "iep", "special education", "esl", "literacy", "numeracy",
    "learning management system", "canvas", "blackboard", "moodle",
    "google classroom", "blended learning", "tutoring", "advising",
    "accreditation", "student outcomes", "formative assessment",
    "instructional coaching", "professional development",
]
_CREATIVE = [
    "graphic design", "visual design", "brand identity", "typography",
    "illustration", "photoshop", "illustrator", "indesign", "after effects",
    "premiere pro", "video editing", "motion graphics", "photography",
    "art direction", "creative direction", "storyboarding", "3d modeling",
    "blender", "cad", "autocad", "solidworks", "revit", "sketchup",
    "prototyping", "design systems", "accessibility", "wcag",
]
_TRADES_SERVICE = [
    "hvac", "electrical", "plumbing", "welding", "carpentry", "machining",
    "cnc", "blueprint reading", "preventive maintenance", "troubleshooting",
    "forklift", "cdl", "commercial driving", "route planning",
    "food safety", "servsafe", "food preparation", "kitchen management",
    "inventory counts", "point of sale", "cash handling", "customer service",
    "scheduling", "shift management", "housekeeping", "groundskeeping",
    "security operations", "dispatch", "first aid",
]

# Named clusters, so a score can say WHERE you fit rather than only how
# much. "62% fit" is a number the user can't act on; "Cloud & Platform 8/9,
# Data & ML 1/6" tells them the role is an infrastructure job with an
# analytics tail they'd have to learn, which is a decision they can make.
# The groups already existed for taxonomy curation -- this just stops
# throwing the grouping away at scoring time.
COMPETENCY_GROUPS: dict[str, list[str]] = {
    # Tech
    "Cloud & platform": _CLOUD_PLATFORM,
    "Languages": _LANGUAGES,
    "Data & ML": _DATA_ML,
    "Databases": _DATABASES,
    "Web & app": _WEB_APP,
    "Practices & tooling": _METHODOLOGY,
    "Security & compliance": _SECURITY_COMPLIANCE,
    # Everyone else. Only the clusters a posting actually mentions are
    # displayed, so a nurse never sees an empty "Cloud & platform" row and a
    # platform engineer never sees "Clinical & patient care".
    "Clinical & patient care": _HEALTHCARE,
    "Legal & regulatory": _LEGAL,
    "Finance & accounting": _FINANCE_ACCOUNTING,
    "Sales & customer": _SALES_CS,
    "Marketing & content": _MARKETING,
    "Operations & supply chain": _OPERATIONS,
    "People & HR": _HR_PEOPLE,
    "Teaching & learning": _EDUCATION,
    "Design & creative": _CREATIVE,
    "Trades & service": _TRADES_SERVICE,
    "Leadership & business": _BUSINESS_SOFT,
}

_TERM_TO_GROUP: dict[str, str] = {
    term.lower(): group
    for group, terms in COMPETENCY_GROUPS.items()
    for term in terms
}

SKILL_KEYWORDS: set[str] = {
    term.lower()
    for group in COMPETENCY_GROUPS.values()
    for term in group
}

# Multi-word terms need their own regex (word-boundary won't span spaces
# correctly otherwise); single-word terms share a simpler path.
_MULTI_WORD = sorted((t for t in SKILL_KEYWORDS if " " in t), key=len, reverse=True)
_SINGLE_WORD = {t for t in SKILL_KEYWORDS if " " not in t}

_ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")
_WORD_RE = re.compile(r"[a-zA-Z0-9+#.]+")


def normalize(text: str) -> str:
    text = text.lower()
    # keep + # . so c++, c#, node.js survive; collapse everything else to spaces
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str) -> set[str]:
    if not text:
        return set()

    normalized = normalize(text)
    found: set[str] = set()

    for term in _MULTI_WORD:
        if term in normalized:
            found.add(term)

    # Strip trailing periods before matching -- "." is kept during normalize()
    # so internal periods survive (node.js), but that also glues a sentence-
    # ending period onto the preceding word (e.g. "ArgoCD." -> "argocd.");
    # rstrip only removes a TRAILING period, so "node.js" (ends in "s") is
    # unaffected while "argocd." correctly becomes "argocd".
    tokens = {t.rstrip(".") for t in _WORD_RE.findall(normalized)}
    found |= tokens & _SINGLE_WORD

    # Bonus: acronyms from the ORIGINAL (case-preserved) text catch tool
    # names not in the static taxonomy (AWS, ETL, CRM, ...) without needing
    # every acronym pre-listed. Lowercased before returning for comparison.
    for match in _ACRONYM_RE.findall(text):
        found.add(match.lower())

    return found


@dataclass
class Competency:
    """One cluster's coverage for this posting."""
    name: str
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    emphasis: int = 0  # how much the posting talks about this cluster

    @property
    def total(self) -> int:
        return len(self.matched) + len(self.missing)

    @property
    def percent(self) -> int:
        return round(100 * len(self.matched) / self.total) if self.total else 0


@dataclass
class MatchResult:
    fit_percent: int
    matched_keywords: list[str] = field(default_factory=list)
    gap_keywords: list[str] = field(default_factory=list)
    jd_keyword_count: int = 0
    # Requirements the posting emphasizes, split by whether the resume
    # evidences them. "5 of 6 core requirements" is a far more useful
    # sentence than a single percentage, because it is the thing a hiring
    # manager is actually screening on.
    core_met: list[str] = field(default_factory=list)
    core_missing: list[str] = field(default_factory=list)
    competencies: list[Competency] = field(default_factory=list)
    # low when the posting had too little extractable text to judge. Without
    # this, a two-line posting mentioning one tool the user happens to have
    # scores 100% and outranks a real match, which is the scorer overstating
    # what it knows.
    confidence: str = "low"

    @property
    def core_total(self) -> int:
        return len(self.core_met) + len(self.core_missing)


# A posting with fewer extractable skill terms than this can't support a
# meaningful score. Chosen because most real postings yield 15-40 terms; in
# the single digits we're reading a stub or a heavily-formatted page.
_CONFIDENT_KEYWORD_COUNT = 12
_MEDIUM_KEYWORD_COUNT = 6


def compute_fit(resume_text: str, jd_text: str, max_gaps: int = 8) -> MatchResult:
    """Emphasis-weighted skills match, grouped into competency clusters.

    Two changes from a plain set intersection, both because the plain
    version answered a question nobody asked:

    1. WEIGHTED BY EMPHASIS. A posting that says "Kubernetes" six times and
       "Jira" once does not weigh them equally, so neither should the score.
       Every term is weighted by how often the posting mentions it, which
       means matching the things the role is actually about moves the number
       more than matching its footnotes. Unweighted, a candidate could miss
       every core requirement, match a dozen incidental tools, and score
       well.

    2. CORE REQUIREMENTS CALLED OUT. Terms the posting leans on repeatedly
       are reported separately from the long tail, because "5 of 6 core
       requirements" is actionable and "62%" is not.

    Still no LLM, still every claim traceable to a term that literally
    appears in both documents -- the point of this scorer is that the user
    can check it, not trust it.
    """
    jd_keywords = extract_keywords(jd_text)
    if not jd_keywords:
        return MatchResult(fit_percent=0, confidence="low")

    resume_keywords = extract_keywords(resume_text)
    matched = jd_keywords & resume_keywords
    gaps = jd_keywords - resume_keywords

    jd_normalized = normalize(jd_text)

    def emphasis(term: str) -> int:
        # At least 1: extract_keywords finds acronyms in the original
        # case-preserved text, which normalize() may have altered, so a
        # genuine term can legitimately count 0 here.
        return max(1, jd_normalized.count(term))

    weights = {term: emphasis(term) for term in jd_keywords}
    total_weight = sum(weights.values())
    matched_weight = sum(weights[t] for t in matched)
    fit_percent = round(100 * matched_weight / total_weight) if total_weight else 0

    matched_ranked = sorted(matched, key=lambda t: weights[t], reverse=True)
    gaps_ranked = sorted(gaps, key=lambda t: weights[t], reverse=True)

    # Core = repeatedly emphasized. Threshold is relative to this posting
    # rather than absolute, since a terse posting mentions everything twice
    # and a verbose one mentions everything six times.
    top_weight = max(weights.values())
    core_cut = max(2, round(top_weight * 0.5))
    core_met = [t for t in matched_ranked if weights[t] >= core_cut]
    core_missing = [t for t in gaps_ranked if weights[t] >= core_cut]

    competencies = []
    for name in COMPETENCY_GROUPS:
        group_matched = [t for t in matched_ranked if _TERM_TO_GROUP.get(t) == name]
        group_missing = [t for t in gaps_ranked if _TERM_TO_GROUP.get(t) == name]
        if not group_matched and not group_missing:
            continue  # posting says nothing about this cluster; don't show an empty row
        competencies.append(Competency(
            name=name,
            matched=group_matched,
            missing=group_missing,
            emphasis=sum(weights[t] for t in group_matched + group_missing),
        ))
    # Ordered by how much the posting cares, so the first row is the one the
    # role is actually about.
    competencies.sort(key=lambda c: c.emphasis, reverse=True)

    if len(jd_keywords) >= _CONFIDENT_KEYWORD_COUNT:
        confidence = "high"
    elif len(jd_keywords) >= _MEDIUM_KEYWORD_COUNT:
        confidence = "medium"
    else:
        confidence = "low"

    return MatchResult(
        fit_percent=fit_percent,
        matched_keywords=matched_ranked,
        gap_keywords=gaps_ranked[:max_gaps],
        jd_keyword_count=len(jd_keywords),
        core_met=core_met,
        core_missing=core_missing,
        competencies=competencies,
        confidence=confidence,
    )


@dataclass
class MarketSignal:
    """One skill, counted across a whole result set."""
    term: str
    postings: int          # how many matched roles asked for it
    total: int             # how many roles were examined
    core_in: int = 0       # of those, how many treated it as a core requirement

    @property
    def share(self) -> float:
        return self.postings / self.total if self.total else 0.0


@dataclass
class MarketReport:
    gaps: list[MarketSignal] = field(default_factory=list)
    strengths: list[MarketSignal] = field(default_factory=list)
    postings_analyzed: int = 0
    low_confidence_postings: int = 0


def analyze_market(matches: list[MatchResult], limit: int = 8, min_share: float = 0.15) -> MarketReport:
    """Roll per-posting results up into what a whole search is telling you.

    A single card can say "this role wants Snowflake and you haven't
    mentioned it". Useful, but weak: one posting is an anecdote and you
    can't tell an outlier requirement from a market-wide one. Across
    twenty-four matched roles the same list becomes a decision --
    "Snowflake appears in 14 of the 24 roles you matched" tells you what to
    learn next, or what your resume is failing to say about work you have
    already done. That is the question a job seeker actually needs answered,
    and no aggregator answers it because none of them have your resume.

    Counted in postings, never averaged into a score: "14 of 24 roles" is
    checkable against the list on screen, and a mean would not be.

    min_share drops one-off mentions. A requirement in a single posting out
    of twenty-four is noise, and presenting it beside a genuine pattern
    would make the panel worth ignoring.

    Low-confidence postings still count -- they are real roles -- but the
    caller is told how many there were, since a report drawn mostly from
    thin postings deserves to be read more loosely.
    """
    real = [m for m in matches if m.jd_keyword_count]
    total = len(real)
    if not total:
        return MarketReport()

    gap_counts: dict[str, int] = {}
    gap_core: dict[str, int] = {}
    strength_counts: dict[str, int] = {}
    strength_core: dict[str, int] = {}

    for m in real:
        # gap_keywords is already truncated for display; use the full
        # difference so the rollup isn't biased by per-card presentation.
        for term in set(m.gap_keywords):
            gap_counts[term] = gap_counts.get(term, 0) + 1
            if term in m.core_missing:
                gap_core[term] = gap_core.get(term, 0) + 1
        for term in set(m.matched_keywords):
            strength_counts[term] = strength_counts.get(term, 0) + 1
            if term in m.core_met:
                strength_core[term] = strength_core.get(term, 0) + 1

    def rank(counts, core_counts):
        signals = [
            MarketSignal(term=t, postings=n, total=total, core_in=core_counts.get(t, 0))
            for t, n in counts.items()
            if n / total >= min_share
        ]
        # Core-requirement weight breaks ties: a term twelve roles list as
        # essential matters more than one twelve roles mention in passing.
        signals.sort(key=lambda s: (s.postings, s.core_in), reverse=True)
        return signals[:limit]

    return MarketReport(
        gaps=rank(gap_counts, gap_core),
        strengths=rank(strength_counts, strength_core),
        postings_analyzed=total,
        low_confidence_postings=sum(1 for m in real if m.confidence == "low"),
    )


def build_cover_letter_opener(job_title: str, matched_keywords: list[str], max_terms: int = 4) -> str:
    """A template-filled opening line, built ONLY from skills the matching
    already verified are on the resume -- no invented claims, no LLM. If
    nothing matched, returns "" (an honest gap gets no fabricated sentence)."""
    terms = matched_keywords[:max_terms]
    if not terms:
        return ""
    if len(terms) == 1:
        skills_phrase = terms[0]
    elif len(terms) == 2:
        skills_phrase = f"{terms[0]} and {terms[1]}"
    else:
        skills_phrase = ", ".join(terms[:-1]) + f", and {terms[-1]}"

    role_phrase = f"the {job_title} role" if job_title else "this role"
    return (
        f"I'm excited to apply for {role_phrase}. My background includes hands-on "
        f"experience with {skills_phrase}, which line up directly with what this "
        f"posting is looking for."
    )
