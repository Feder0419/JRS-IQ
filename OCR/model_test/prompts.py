"""
The prompts for the two ways of reading a report. They share one answer format so the
results can be scored the same way. Edit the wording here (not elsewhere) to match how
your model is normally prompted, but keep BOTH arms equally careful: the test is only
fair if the only difference is what image(s) the model sees.
"""

# What to read for each field, and the printed label/caption the value is found next to.
FIELD_INFO = {
    "form_number": "the form number printed in the title, e.g. 'H-3', 'P-2', 'P-3', 'U-1A', '154-A' (label: the word FORM)",
    "report_date": "the date written next to the label 'Date' above the manufacturer's signature (not the certificate-expiry date, not the inspector's date lower on the page)",
    "year_built": "the year built (label: 'Year Built' or 'Yr. Built')",
    "boiler_no": "the boiler / vessel number (label: 'Boiler No.', 'Vessel No.' or 'ID Nos.')",
    "serial_no": "the manufacturer's serial number (printed above the caption '(Mfrs. Serial No.)'; often the same as the boiler number)",
    "national_board_no": "the National Board number (printed above the caption '(Natl. Board No.)', 'Natl. Bd. No.' or \"(Nat'l Brd No.)\")",
    "manufactured_by": "name and address after 'Manufactured by'",
    "manufactured_for": "name and address after 'Manufactured for' (the purchaser). It can be blank on the form",
    "material_spec": "the material specification of the shell / tube sheet, e.g. 'SA-285-C', 'SA-516-70', 'SA285C' (label: 'Material spec. no.', 'Tube sheet', 'Shell: Material')",
}

ANSWER_FORMAT = (
    'Return ONLY JSON in this shape, one entry per field asked for:\n'
    '{"fields": {"<field_name>": {"value": "<text exactly as written, or null>", '
    '"label_seen": "<the printed label or caption the value sits next to, or null>", '
    '"confidence": "high" | "medium" | "low"}}}'
)

COMMON_RULES = (
    "- Copy each value exactly as printed or written (keep hyphens, slashes, punctuation, spelling). Do not correct or reformat it.\n"
    "- If you cannot read a value, or it is not there, use null. Do not guess.\n"
    "- label_seen is the printed label or caption you can actually see next to the value; use null if you cannot see one.\n"
)


def _field_lines(fields):
    return "\n".join(f"- {f}: {FIELD_INFO[f]}" for f in fields)


def full_page_prompt(fields, n_pages: int) -> str:
    return (
        "These are scanned ASME boiler / pressure-vessel data reports (typed and handwritten, often faded). "
        f"The {n_pages} attached image(s) are the page(s) of ONE report, in order.\n\n"
        f"Read these fields:\n{_field_lines(fields)}\n\nRules:\n{COMMON_RULES}\n{ANSWER_FORMAT}"
    )


def snippets_prompt(fields) -> str:
    order = "\n".join(f"- image {i + 1}: {f}" for i, f in enumerate(fields))
    return (
        "These are scanned ASME boiler / pressure-vessel data reports (typed and handwritten, often faded). "
        "Each attached image is a SMALL CROP from one report, cut around a single field. "
        f"The images, in order, are for:\n{order}\n\n"
        f"What to read:\n{_field_lines(fields)}\n\nRules:\n{COMMON_RULES}"
        "- Read each field from its own image only. If that image does not show the field's printed label or caption "
        "next to a value, answer value null and label_seen null (the page will be read in full instead).\n\n"
        f"{ANSWER_FORMAT}"
    )
