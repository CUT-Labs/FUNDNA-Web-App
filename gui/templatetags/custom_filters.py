from django import template

register = template.Library()


@register.filter
def weighted_score(score_tuple):
    try:
        score, weight = score_tuple
        if score is not None and weight is not None:
            return f"{(score * weight / 100):.2f}"
        else:
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"


@register.filter
def score_and_rank(score_tuple):
    try:
        score, by = score_tuple
        if score is not None and by is not None:
            # Format score to two decimal places and join design names with commas
            by_designs = ", ".join(str(d) for d in by)
            return f"{score:.2f} by [{by_designs}]"
        else:
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"


@register.filter
def format_meta_cell(value):
    """
    Format cells in meta ranking table: apply two decimal places to floats,
    and return strings as-is.
    """
    try:
        # Attempt to cast to float, format to two decimal places if successful
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        # If casting fails (e.g., for strings), return the value as-is
        return value
