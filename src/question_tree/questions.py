"""The wording of every question. Each entry is (text, allowed answers).

A text may use the columns of the image's csv row in braces, for example {product_title}.
"""

YES_NO = ["yes", "no"]
SYSTEM = "You answer one question about a product image of an online shop."
LISTING = "The listing is for: {product_title} ({product_type}). "


def multiple_choice(question, options):
    """A question with one line of explanation per class."""
    text = question + " Options:\n" + "\n".join(f"- {name}: {explanation}" for name, explanation in options)
    return text, [name for name, _ in options]


QUESTIONS = {
    # ---- subject
    "energy_label": ("Is this image an EU energy efficiency label, the chart with colored arrows from A to G?", YES_NO),
    "placeholder": ("Is this a placeholder picture that only says that no image is available?", YES_NO),
    "logo": ("Does this image show only a brand or company logo and nothing else?", YES_NO),
    "logo2": ("Is this image only the logo of the brand or company, and nothing else? A certification seal, eco-label, award badge, recycling symbol or pictogram is not a brand logo.", YES_NO),
    "mark": ("Is this image only a certification seal, eco-label, award badge, quality mark, recycling symbol or pictogram, without a product photo?", YES_NO),
    "photo": ("Does this image contain a real photo or a realistic 3D rendering of a physical product?", YES_NO),
    "infographic": ("Is this image a graphic without a real product photo, for example a text page, a table, a chart, a diagram, a line drawing or a collage of icons?", YES_NO),
    "textonly": ("Does this image consist only of written text or a table, such as an ingredient list, a description, quotes or a data sheet, without any picture of the product?", YES_NO),
    "packaging": ("Is a retail box, carton or blister pack visible in the image?", YES_NO),
    "product_outside": ("Is the actual product visible as a real object next to its box, or through a window or a clear blister, and not only printed on the box?", YES_NO),
    "box_only": ("Is the image only a closed retail box or carton? A picture of the product printed on the box does not make the product visible.", YES_NO),
    "pack_and_real": ("Do you see a retail box and, in addition, the real physical product outside the box or through a clear window? A picture of the product printed on the box does not count.", YES_NO),
    "edible": ("The listing is for: {product_title} ({product_type}, {category}). Is this something to eat or drink, pet food or a dietary supplement?", YES_NO),
    "several": ("Does the image show more than one separate object?", YES_NO),
    "identical": ("Are several identical copies of the same item shown, all of the same size and all facing the same way?", YES_NO),
    "variants": ("Are several versions of the same product shown that differ in color or in size?", YES_NO),
    "views2": ("Is the very same single item shown two or more times in this image, for example once from the front and once from the back, or once open and once closed? Several different items, or several copies in the same position, do not count.", YES_NO),
    "scope": ("Is one main product shown together with its accessories, parts or cables laid out next to it?", YES_NO),
    "scope2": ("Does the image show one main product plus at least two separate loose items that come with it, such as cables, attachments, a charger or a manual, laid out side by side? A single product with built-in parts does not count.", YES_NO),
    "part_of": ("Is the object in this image a part, an accessory or an attachment that only works together with a bigger product?", YES_NO),
    "subject_mc": multiple_choice("What does the image show? Think of what the customer buys.", [
        ("full_product", "the sold item, whole, nothing cut off (a food pack counts as the product)"),
        ("detail_closeup", "a zoomed part, texture or swatch, or the product cut off at the frame"),
        ("accessories", "only an accessory or peripheral, not the main product"),
        ("packaging", "only the outer box, carton or bag, the product itself not visible"),
        ("package_and_product", "the product and its packaging both visible"),
        ("multipack_trade_unit", "several identical units in the same orientation"),
        ("delivery_scope", "the main product together with what comes with it (parts, cables, accessories)"),
        ("product_grouping", "several variants of the product (colors or sizes)"),
        ("multiview", "one product shown two or more times from different angles or states"),
        ("energy_label", "the EU energy efficiency label (A to G)"),
        ("infographic", "no product photo: text page, table, chart, diagram, icons, line drawing"),
        ("logo", "a single brand or company logo, nothing else"),
        ("placeholder", "a generic 'image coming soon' icon"),
    ]),
    "subject_mc2": multiple_choice(LISTING + "What does the image show?", [
        ("full_product", "the listed product itself, complete, as one item. Food, drinks and supplements in their own bag, jar, bottle or box count as full_product, also when seen from the back"),
        ("detail_closeup", "only a part of the product is visible because it is zoomed in or cut off by the image border; also a texture or material sample, or the inner pages of a book"),
        ("accessories", "the object shown is not the listed product itself but an add-on or part for it, such as a cable, charger, case, remote control, attachment or spare part"),
        ("packaging", "only a closed retail box or carton. A picture of the product printed on the box still counts as packaging"),
        ("package_and_product", "a retail box and also the real physical product next to it or visible through a clear window"),
        ("multipack_trade_unit", "several identical copies of the product in the same position, such as a six-pack or a tray"),
        ("delivery_scope", "the product laid out together with the separate items that come with it, such as cables, attachments and manuals"),
        ("product_grouping", "the same product in several colors or sizes side by side"),
        ("multiview", "the very same single item shown two or more times, for example front and back, or open and closed"),
        ("energy_label", "the EU energy efficiency label with colored arrows from A to G"),
        ("infographic", "no product photo: only text, a table, a chart, a drawing, icons, or a certification seal or eco-label on its own"),
        ("logo", "only the logo of a brand or company. A seal, eco-label or icon is not a logo"),
        ("placeholder", "a generic picture that says no image is available"),
    ]),
    # ---- individuals
    "person": ("Is a real person, a real hand or a real animal visible in the photo? People printed on a box or shown on a screen do not count.", YES_NO),
    # ---- text overlay
    "nutrition": ("Is a nutrition table with values such as energy, fat and protein visible?", YES_NO),
    "nutrition2": ("Is a nutrition facts table visible that lists energy in kJ or kcal together with fat, carbohydrates and protein as numbers?", YES_NO),
    "dims": ("Are dimension lines, measurement arrows or size numbers drawn on top of the image to show how big the product is?", YES_NO),
    "promo": ("Is there advertising text placed on the image as a graphic layer, such as a slogan, a headline, a list of benefits or a sale badge? Text printed on the product or its box does not count.", YES_NO),
    "seal": ("Is there a certification seal, a test winner badge, an award logo or an eco-label placed on the image as a graphic, separate from the product itself?", YES_NO),
    "added_layer": ("Look at the text in this image. Was it placed on top of the photo as a separate graphic layer, like a caption, a headline or a banner (answer added)? Or is it only printed on the product or its packaging (answer printed)? If there is no text, answer none.", ["added", "printed", "none"]),
    "overlay_mc": multiple_choice("Which kind of text or graphic element was added digitally on top of the photo? Text printed on the product or pack does not count, except a nutrition table.", [
        ("nutritional_info", "a nutrition table (energy, fat, protein per 100 g, or similar)"),
        ("technical_specs", "dimension lines, measurements, size tags, callouts to parts"),
        ("promotional_text", "marketing claims, slogans, sale banners, 'NEW' badges"),
        ("certification_seal", "trust marks, certificates, eco-labels, award badges"),
        ("other", "other added elements: icons, pictograms, stickers, watermarks"),
        ("none", "nothing added on top"),
    ]),
    "overlay_mc2": multiple_choice("Was anything added on top of the photo as a graphic layer? Text that is printed on the product or on its packaging does not count as added.", [
        ("nutritional_info", "a nutrition table with energy, fat, carbohydrates and protein is visible. This one also counts when it is printed on the pack"),
        ("technical_specs", "added dimension lines, measurements, technical values such as temperatures or wattage, or labels pointing at parts"),
        ("promotional_text", "added advertising text such as a slogan, a headline, a list of benefits or a NEW or SALE badge"),
        ("certification_seal", "an added test seal, award badge, quality mark or eco-label, or the image is only such a seal"),
        ("other", "added icons, pictograms, arrows or stickers without advertising text"),
        ("none", "nothing was added; any text is printed on the product or the pack itself"),
    ]),
    # ---- background
    "scene": ("Is the product shown in a real environment, for example in a room, outdoors, on a table or on the floor, or being used or held by a person?", YES_NO),
    "plain": ("Is the background behind the product a single plain color or plain white, without any objects, furniture or scenery?", YES_NO),
    "bg_mc": multiple_choice("What is behind the product?", [
        ("white", "pure white, a clean cutout, or no backdrop visible at all"),
        ("color", "a plain uniform backdrop that is not white: grey, cream, black, any color or gradient"),
        ("scene_lifestyle", "a real environment: a room, outdoors, a table or surface, props, a person"),
    ]),
    # ---- language
    "words2": ("Ignore brand names, product names, model numbers, single technical words and numbers. Is there at least one full phrase or sentence in the image, such as a claim, a description, an instruction or an ingredient list?", YES_NO),
    "lang": ("In which language are most of the words in this image written? Ignore brand names, model names, numbers and technical terms. Answer none if there are no other words.", ["German", "English", "French", "Italian", "Dutch", "other", "none"]),
    # ---- perspective, only used when the contour model is not available
    "bottom": ("Is the camera looking straight up at the underside of the product?", YES_NO),
    "persp_mc": multiple_choice("Find the front of the product. Seen from above with the camera at 6 o'clock, where does the front point?", [
        ("front_head_on", "straight at the camera, at product height"),
        ("front_angle_left", "toward the bottom left (7 to 8 o'clock)"),
        ("front_angle_right", "toward the bottom right (4 to 5 o'clock)"),
        ("back_head_on", "away from the camera, we see the back straight on"),
        ("back_angle_left", "toward the top right, we see the back (1 to 2 o'clock)"),
        ("back_angle_right", "toward the top left, we see the back (10 to 11 o'clock)"),
        ("left_side", "straight left, a pure side profile (9 o'clock)"),
        ("right_side", "straight right, a pure side profile (3 o'clock)"),
        ("top_down", "the camera looks straight down onto the top"),
        ("bottom_up", "the camera looks straight up at the underside"),
    ]),
}

# What the final tree asks: 22 yes/no questions and 3 multiple-choice questions per image.
TREE_YES_NO = [
    "energy_label", "placeholder", "logo2", "mark", "photo", "infographic", "textonly", "pack_and_real", "box_only",
    "packaging", "variants", "identical", "several", "scope", "scene", "plain", "person", "nutrition", "dims",
    "promo", "seal", "words2",
]
TREE_MULTIPLE_CHOICE = ["subject_mc", "overlay_mc", "lang"]
TREE_QUESTIONS = TREE_YES_NO + TREE_MULTIPLE_CHOICE

# Grouping. These seven change their answer too often when they share a request, so they are always asked alone.
ASKED_ALONE = ["packaging", "pack_and_real", "box_only", "nutrition", "mark", "energy_label", "scene"]
ASKED_TOGETHER = [key for key in TREE_YES_NO if key not in ASKED_ALONE]
