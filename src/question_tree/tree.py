"""The decision tree: from the answers of one image to its labels.

The first version of the tree was written after looking at single questions. Eleven sharper rules were then
written from the cases it got wrong. Each one is a switch here, so its effect can be measured on its own.
Six of them improved the score on the tuning images and are part of the final tree.
"""

SHARPENED_RULES = {
    "no_fallback_to_multiple_choice": "pack-plus-product and multiview are no longer taken from the multiple-choice answer",
    "seal_is_not_a_logo": "logo: seals, eco-labels and icons are not brand logos",
    "lone_seal_or_text_is_infographic": "infographic: also a seal on its own or an image that is only text",
    "printed_product_is_packaging": "packaging: a closed box, also with the product printed on it; pack-plus-product needs the real item",
    "food_pack_is_the_product": "food, drinks and supplements in their own pack are the product",
    "scope_needs_loose_items": "delivery scope: the main product plus at least two loose items",
    "multiview_is_the_same_item": "multiview: the very same item shown twice",
    "strict_nutrition_table": "nutrition: a table with energy, fat, carbohydrates and protein as numbers",
    "lone_seal_is_certification": "a seal on its own is the overlay class certification_seal",
    "promotion_is_an_added_layer": "promotional text must be an added layer, not printed on the pack",
    "no_phrase_means_no_language": "language: none when the image holds no full phrase, brand names do not count",
}
KEPT_RULES = frozenset({
    "no_fallback_to_multiple_choice", "seal_is_not_a_logo", "lone_seal_or_text_is_infographic",
    "printed_product_is_packaging", "lone_seal_is_certification", "no_phrase_means_no_language",
})
NO_RULES = frozenset()


def classify(answers, white, rules=KEPT_RULES):
    """The labels of one image.

    answers: {question key: answer}. white: the pixel measurement of the backdrop. rules: the sharpened rules in use.
    The perspective returned here is the answer of Apertus. The contour model replaces it in the full pipeline.
    """
    yes = lambda key: answers.get(key) == "yes"  # noqa: E731
    no = lambda key: answers.get(key) == "no"  # noqa: E731
    use = lambda rule: rule in rules  # noqa: E731
    multiple_choice = answers.get("subject_mc")
    food = use("food_pack_is_the_product") and yes("edible")

    # subject: what the image shows
    if yes("energy_label"):
        subject = "energy_label"
    elif yes("placeholder"):
        subject = "placeholder"
    elif (yes("logo2") if use("seal_is_not_a_logo") else yes("logo")) and no("photo"):
        subject = "logo"
    elif no("photo") and (yes("infographic") or (use("lone_seal_or_text_is_infographic") and (yes("mark") or yes("textonly")))):
        subject = "infographic"
    elif use("printed_product_is_packaging") and not food and yes("pack_and_real"):
        subject = "package_and_product"
    elif use("printed_product_is_packaging") and not food and (yes("box_only") or yes("packaging")):
        subject = "packaging"
    elif not use("printed_product_is_packaging") and not food and yes("packaging"):
        subject = "package_and_product" if yes("product_outside") else "packaging"
    elif yes("variants"):
        subject = "product_grouping"
    elif use("multiview_is_the_same_item") and yes("views2") and yes("several"):
        subject = "multiview"
    elif yes("identical") and yes("several"):
        subject = "multipack_trade_unit"
    elif yes("scope2") if use("scope_needs_loose_items") else (yes("scope") and yes("several")):
        subject = "delivery_scope"
    elif multiple_choice == "detail_closeup":
        subject = "detail_closeup"
    elif not use("no_fallback_to_multiple_choice") and not food and multiple_choice in ("package_and_product", "multiview"):
        subject = multiple_choice
    else:
        subject = "full_product"

    # background: Apertus decides scene or not, the pixel measurement decides white or color
    if yes("scene"):
        background = "scene_lifestyle"
    elif white:
        background = "white"
    elif no("plain"):
        background = "scene_lifestyle"
    else:
        background = "color"

    # text overlay
    overlay_choice = answers.get("overlay_mc")
    if yes("nutrition2") if use("strict_nutrition_table") else yes("nutrition"):
        overlay = "nutritional_info"
    elif use("lone_seal_is_certification") and yes("mark"):
        overlay = "certification_seal"
    elif yes("dims"):
        overlay = "technical_specs"
    elif yes("promo") and (answers.get("added_layer") == "added" if use("promotion_is_an_added_layer") else overlay_choice == "promotional_text"):
        overlay = "promotional_text"
    elif yes("seal") and overlay_choice in ("certification_seal", "other"):
        overlay = "certification_seal"
    else:
        overlay = "none"

    language = "none" if (use("no_phrase_means_no_language") and no("words2")) else (answers.get("lang") or "none")
    perspective = "bottom_up" if yes("bottom") else (answers.get("persp_mc") or "front_head_on")
    return {
        "subject": subject,
        "background": background,
        "perspective": perspective,
        "individuals": "individual_present" if yes("person") else "none",
        "text_classification": overlay,
        "text_language": language,
    }


def multiple_choice_only(answers):
    """The simpler method the tree is compared with: one multiple-choice question per dimension."""
    return {
        "subject": answers.get("subject_mc"),
        "background": answers.get("bg_mc"),
        "perspective": answers.get("persp_mc"),
        "individuals": "individual_present" if answers.get("person") == "yes" else "none",
        "text_classification": answers.get("overlay_mc"),
        "text_language": answers.get("lang"),
    }
