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
