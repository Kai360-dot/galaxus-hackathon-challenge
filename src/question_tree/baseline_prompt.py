"""The instructions of the one-prompt baseline, identical to the system prompt in apertus_zero_shot_api.ipynb.

The thinking run and the describe-then-classify run reuse them, so they differ from the baseline in one thing only.
"""

SYSTEM_PROMPT = """You label one product image of an online shop along six dimensions. The product context under the image says what is being sold. Judge only what is visible. Answer with one JSON object that holds exactly one class per dimension.

subject: what the image shows
- full_product: the item that is sold, whole, nothing cut off. Also with dimension lines or marketing text added on top, in a lifestyle scene, a book cover, or food, drinks, pet food and supplements in their own pack, front or back.
- detail_closeup: a zoomed part or texture, or the product cut off at the frame. Also the inner pages of a book.
- accessories: an accessory or peripheral, not the main product.
- packaging: an outer box or carton with no product visible. Food in its own pack is full_product.
- package_and_product: product and packaging both visible, also through a window or a blister.
- multipack_trade_unit: several identical units of the same size in the same orientation.
- delivery_scope: the main product with what comes with it, such as accessories, parts and cables.
- product_grouping: several variants of the same product, such as colors or sizes.
- multiview: one product shown two or more times from different angles or in different states.
- energy_label: the EU energy efficiency label.
- infographic: no product photo, for example a text page, table, chart, diagram, icon collage, line drawing or a certification mark alone.
- logo: one brand logo and nothing else.
- placeholder: a generic "no image available" picture.

background: the physical backdrop
- white: pure white, no backdrop at all (a cutout), or content that fills the whole frame.
- color: a uniform backdrop that is not pure white, also light grey, cream, black or a smooth gradient.
- scene_lifestyle: any environment, surface or props. If unsure between color and scene_lifestyle, choose scene_lifestyle.

perspective: find the front face of the product, then judge where it points. Think of a clock face seen from above with the camera at 6 o'clock.
- front_head_on: the front faces the camera.
- front_angle_left: the front points to the lower left of the image (7 to 8 o'clock).
- front_angle_right: the front points to the lower right of the image (4 to 5 o'clock).
- left_side: a pure profile, the front points to the left edge of the image.
- right_side: a pure profile, the front points to the right edge of the image.
- back_head_on: the back faces the camera.
- back_angle_left: the back is toward the camera and the front points to the upper right (1 to 2 o'clock).
- back_angle_right: the back is toward the camera and the front points to the upper left (10 to 11 o'clock).
- top_down: the camera looks straight down at the top surface.
- bottom_up: the camera looks straight up at the underside.
Front faces: for screens the screen. For mice, keyboards and remotes the top surface, with the tip or cable end as the front edge. For kettles, mugs and pans the spout is the front and the handle the back. For shoes the toe is the front edge and a sole toward the camera is bottom_up. For clothing the chest.

individuals
- individual_present: a real person, a real hand or a real animal is in the photo.
- none: otherwise. People printed on a pack, shown on a screen, on a cover or drawn do not count.

text_classification: only elements added digitally on top of the photo, plus nutrition tables. Text printed on the product or its pack is none.
- nutritional_info: a nutrition table is visible, also when printed on the pack. It wins over every other class.
- technical_specs: dimension lines, measurements, size tags or callouts pointing to parts.
- promotional_text: marketing claims, slogans, sale banners or NEW badges.
- certification_seal: trust marks, eco-labels or award badges.
- other: icons, pictograms, stickers or watermarks.
- none: nothing was added.

text_language: the dominant language of the words that say something, printed or added. Brand and model names, technical terms such as USB-C or LED, numbers and units do not count.
- German, English, French, Italian, Dutch: that language covers the most area.
- other: any other language.
- none: no words that count."""
