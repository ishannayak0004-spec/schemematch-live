"""Transparent outreach rules for individual household screening."""
QUESTIONS = {
    "unbanked": "Does any adult lack a bank account?",
    "woman": "Is there an adult woman in the household?",
    "poor": "Does the household identify as poor for the Ujjwala deprivation declaration?",
    "no_lpg": "Is the household without an existing LPG connection?",
    "land": "Does the farmer family own cultivable land?"
}
SCHEMES = [
    dict(name="PM Jan Dhan Yojana", category="Banking", needs=["unbanked"],
         reason="An adult needs access to a basic bank account.",
         url="https://financialservices.gov.in/node/3463"),
    dict(name="PM Ujjwala Yojana", category="Clean cooking", needs=["woman","poor","no_lpg"],
         reason="An adult woman in a self-reported poor household lacks a household LPG connection.",
         url="https://www.pmuy.gov.in/about.html"),
    dict(name="PM-KISAN", category="Agriculture", needs=["land"],
         reason="The family reports owning cultivable farmland. Verify land records and exclusions.",
         url="https://pmkisan.gov.in/")
]


def screen(answers):
    if not isinstance(answers, dict) or any(type(answers.get(q)) is not bool for q in QUESTIONS):
        raise ValueError("Answer each of the five questions.")
    return [s for s in SCHEMES if all(answers[key] for key in s["needs"])]
