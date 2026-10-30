| Condition | n | Success | Wilson 95% | Paired vs E0 (E0-only / cond-only, p) | Wrong object lifted first | Outcomes |
|---|---:|---:|---|---|---:|---|
| OBJ-E0 | 9 | 9/9 | [0.70, 1.00] | — | 0 | success 9 |
| OBJ-swap | 9 | 0/9 | [0.00, 0.30] | 9 / 0, p=0.00391 | 7 | no_lift 2, wrong_object_first 7 |
| OBJ-swap-ctrl | 9 | 7/9 | [0.45, 0.94] | 2 / 0, p=0.5 | 0 | no_lift 1, success 7, target_lifted_not_placed 1 |
| OBJ-lang | 9 | 8/9 | [0.56, 0.98] | 1 / 0, p=1.0 | 0 | success 8, target_lifted_not_placed 1 |
| GOAL-E0 | 9 | 9/9 | [0.70, 1.00] | — | 0 | success 9 |
| GOAL-move | 9 | 4/9 | [0.19, 0.73] | 5 / 0, p=0.0625 | 0 | success 4, target_lifted_not_placed 5 |
| GOAL-move-ctrl | 8 | 8/8 | [0.68, 1.00] | 0 / 0, p=1.0 | 0 | success 8 |
| LONG-E0 | 9 | 9/9 | [0.70, 1.00] | — | 0 | success 9 |
| LONG-pre | 9 | 3/9 | [0.12, 0.65] | 6 / 0, p=0.03125 | 4 | remaining_subgoal_not_done 6, success 3 |
| LONG-undo | 9 | 0/9 | [0.00, 0.30] | 9 / 0, p=0.00391 | 0 | returned_subgoal_not_redone 9 |

| Episode | Prompt | Target | Success | Steps | First lifted | Outcome |
|---|---|---|---|---:|---|---|
| GOAL-E0-t01-s00 | put the bowl on the stove | akita_black_bowl_1 | True | 78 | akita_black_bowl_1 | success |
| GOAL-E0-t01-s01 | put the bowl on the stove | akita_black_bowl_1 | True | 83 | akita_black_bowl_1 | success |
| GOAL-E0-t01-s02 | put the bowl on the stove | akita_black_bowl_1 | True | 79 | akita_black_bowl_1 | success |
| GOAL-E0-t06-s00 | put the cream cheese in the bowl | cream_cheese_1 | True | 119 | cream_cheese_1 | success |
| GOAL-E0-t06-s01 | put the cream cheese in the bowl | cream_cheese_1 | True | 89 | cream_cheese_1 | success |
| GOAL-E0-t06-s02 | put the cream cheese in the bowl | cream_cheese_1 | True | 92 | cream_cheese_1 | success |
| GOAL-E0-t08-s00 | put the bowl on the plate | akita_black_bowl_1 | True | 80 | akita_black_bowl_1 | success |
| GOAL-E0-t08-s01 | put the bowl on the plate | akita_black_bowl_1 | True | 76 | akita_black_bowl_1 | success |
| GOAL-E0-t08-s02 | put the bowl on the plate | akita_black_bowl_1 | True | 80 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t01-s00 | put the bowl on the stove | akita_black_bowl_1 | True | 79 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t01-s01 | put the bowl on the stove | akita_black_bowl_1 | True | 91 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t01-s02 | put the bowl on the stove | akita_black_bowl_1 | True | 80 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t06-s01 | put the cream cheese in the bowl | cream_cheese_1 | True | 107 | cream_cheese_1 | success |
| GOAL-move-ctrl-t06-s02 | put the cream cheese in the bowl | cream_cheese_1 | True | 104 | cream_cheese_1 | success |
| GOAL-move-ctrl-t08-s00 | put the bowl on the plate | akita_black_bowl_1 | True | 84 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t08-s01 | put the bowl on the plate | akita_black_bowl_1 | True | 79 | akita_black_bowl_1 | success |
| GOAL-move-ctrl-t08-s02 | put the bowl on the plate | akita_black_bowl_1 | True | 83 | akita_black_bowl_1 | success |
| GOAL-move-t01-s00 | put the bowl on the stove | akita_black_bowl_1 | True | 78 | akita_black_bowl_1 | success |
| GOAL-move-t01-s01 | put the bowl on the stove | akita_black_bowl_1 | True | 68 | akita_black_bowl_1 | success |
| GOAL-move-t01-s02 | put the bowl on the stove | akita_black_bowl_1 | True | 104 | akita_black_bowl_1 | success |
| GOAL-move-t06-s00 | put the cream cheese in the bowl | cream_cheese_1 | False | 300 | cream_cheese_1 | target_lifted_not_placed |
| GOAL-move-t06-s01 | put the cream cheese in the bowl | cream_cheese_1 | False | 300 | cream_cheese_1 | target_lifted_not_placed |
| GOAL-move-t06-s02 | put the cream cheese in the bowl | cream_cheese_1 | False | 300 | cream_cheese_1 | target_lifted_not_placed |
| GOAL-move-t08-s00 | put the bowl on the plate | akita_black_bowl_1 | False | 300 | akita_black_bowl_1 | target_lifted_not_placed |
| GOAL-move-t08-s01 | put the bowl on the plate | akita_black_bowl_1 | True | 81 | akita_black_bowl_1 | success |
| GOAL-move-t08-s02 | put the bowl on the plate | akita_black_bowl_1 | False | 300 | akita_black_bowl_1 | target_lifted_not_placed |
| LONG-E0-t00-s00 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | True | 260 | alphabet_soup_1 | success |
| LONG-E0-t00-s01 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | True | 254 | alphabet_soup_1 | success |
| LONG-E0-t00-s02 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | True | 249 | alphabet_soup_1 | success |
| LONG-E0-t01-s00 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | True | 259 | cream_cheese_1 | success |
| LONG-E0-t01-s01 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | True | 262 | cream_cheese_1 | success |
| LONG-E0-t01-s02 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | True | 242 | cream_cheese_1 | success |
| LONG-E0-t07-s00 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | True | 230 | alphabet_soup_1 | success |
| LONG-E0-t07-s01 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | True | 279 | alphabet_soup_1 | success |
| LONG-E0-t07-s02 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | True | 253 | alphabet_soup_1 | success |
| LONG-pre-t00-s00 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | False | 520 | None | remaining_subgoal_not_done |
| LONG-pre-t00-s01 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | True | 443 | ketchup_1 | success |
| LONG-pre-t00-s02 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | True | 446 | orange_juice_1 | success |
| LONG-pre-t01-s00 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | milk_1 | remaining_subgoal_not_done |
| LONG-pre-t01-s01 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | None | remaining_subgoal_not_done |
| LONG-pre-t01-s02 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | None | remaining_subgoal_not_done |
| LONG-pre-t07-s00 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | True | 111 | cream_cheese_1 | success |
| LONG-pre-t07-s01 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | False | 520 | None | remaining_subgoal_not_done |
| LONG-pre-t07-s02 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | False | 520 | ketchup_1 | remaining_subgoal_not_done |
| LONG-undo-t00-s00 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| LONG-undo-t00-s01 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| LONG-undo-t00-s02 | put both the alphabet soup and the tomato sauce in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| LONG-undo-t01-s00 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | cream_cheese_1 | returned_subgoal_not_redone |
| LONG-undo-t01-s01 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | cream_cheese_1 | returned_subgoal_not_redone |
| LONG-undo-t01-s02 | put both the cream cheese box and the butter in the basket | cream_cheese_1 | False | 520 | cream_cheese_1 | returned_subgoal_not_redone |
| LONG-undo-t07-s00 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| LONG-undo-t07-s01 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| LONG-undo-t07-s02 | put both the alphabet soup and the cream cheese box in the basket | alphabet_soup_1 | False | 520 | alphabet_soup_1 | returned_subgoal_not_redone |
| OBJ-E0-t00-s00 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 212 | alphabet_soup_1 | success |
| OBJ-E0-t00-s01 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 137 | alphabet_soup_1 | success |
| OBJ-E0-t00-s02 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 144 | alphabet_soup_1 | success |
| OBJ-E0-t01-s00 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 133 | cream_cheese_1 | success |
| OBJ-E0-t01-s01 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 126 | cream_cheese_1 | success |
| OBJ-E0-t01-s02 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 113 | cream_cheese_1 | success |
| OBJ-E0-t04-s00 | pick up the ketchup and place it in the basket | ketchup_1 | True | 146 | ketchup_1 | success |
| OBJ-E0-t04-s01 | pick up the ketchup and place it in the basket | ketchup_1 | True | 135 | ketchup_1 | success |
| OBJ-E0-t04-s02 | pick up the ketchup and place it in the basket | ketchup_1 | True | 149 | ketchup_1 | success |
| OBJ-lang-t00-s00 | pick up the salad dressing and place it in the basket | salad_dressing_1 | True | 106 | salad_dressing_1 | success |
| OBJ-lang-t00-s01 | pick up the salad dressing and place it in the basket | salad_dressing_1 | True | 108 | salad_dressing_1 | success |
| OBJ-lang-t00-s02 | pick up the salad dressing and place it in the basket | salad_dressing_1 | True | 119 | salad_dressing_1 | success |
| OBJ-lang-t01-s00 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 135 | alphabet_soup_1 | success |
| OBJ-lang-t01-s01 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | False | 280 | alphabet_soup_1 | target_lifted_not_placed |
| OBJ-lang-t01-s02 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 151 | alphabet_soup_1 | success |
| OBJ-lang-t04-s00 | pick up the bbq sauce and place it in the basket | bbq_sauce_1 | True | 117 | bbq_sauce_1 | success |
| OBJ-lang-t04-s01 | pick up the bbq sauce and place it in the basket | bbq_sauce_1 | True | 141 | bbq_sauce_1 | success |
| OBJ-lang-t04-s02 | pick up the bbq sauce and place it in the basket | bbq_sauce_1 | True | 126 | bbq_sauce_1 | success |
| OBJ-swap-ctrl-t00-s00 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 152 | alphabet_soup_1 | success |
| OBJ-swap-ctrl-t00-s01 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | False | 280 | alphabet_soup_1 | target_lifted_not_placed |
| OBJ-swap-ctrl-t00-s02 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | True | 132 | alphabet_soup_1 | success |
| OBJ-swap-ctrl-t01-s00 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 130 | cream_cheese_1 | success |
| OBJ-swap-ctrl-t01-s01 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 130 | cream_cheese_1 | success |
| OBJ-swap-ctrl-t01-s02 | pick up the cream cheese and place it in the basket | cream_cheese_1 | True | 108 | cream_cheese_1 | success |
| OBJ-swap-ctrl-t04-s00 | pick up the ketchup and place it in the basket | ketchup_1 | True | 153 | ketchup_1 | success |
| OBJ-swap-ctrl-t04-s01 | pick up the ketchup and place it in the basket | ketchup_1 | False | 280 | None | no_lift |
| OBJ-swap-ctrl-t04-s02 | pick up the ketchup and place it in the basket | ketchup_1 | True | 136 | ketchup_1 | success |
| OBJ-swap-t00-s00 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | False | 280 | salad_dressing_1 | wrong_object_first |
| OBJ-swap-t00-s01 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | False | 280 | salad_dressing_1 | wrong_object_first |
| OBJ-swap-t00-s02 | pick up the alphabet soup and place it in the basket | alphabet_soup_1 | False | 280 | salad_dressing_1 | wrong_object_first |
| OBJ-swap-t01-s00 | pick up the cream cheese and place it in the basket | cream_cheese_1 | False | 280 | alphabet_soup_1 | wrong_object_first |
| OBJ-swap-t01-s01 | pick up the cream cheese and place it in the basket | cream_cheese_1 | False | 280 | alphabet_soup_1 | wrong_object_first |
| OBJ-swap-t01-s02 | pick up the cream cheese and place it in the basket | cream_cheese_1 | False | 280 | alphabet_soup_1 | wrong_object_first |
| OBJ-swap-t04-s00 | pick up the ketchup and place it in the basket | ketchup_1 | False | 280 | None | no_lift |
| OBJ-swap-t04-s01 | pick up the ketchup and place it in the basket | ketchup_1 | False | 280 | None | no_lift |
| OBJ-swap-t04-s02 | pick up the ketchup and place it in the basket | ketchup_1 | False | 280 | bbq_sauce_1 | wrong_object_first |

Excluded before policy start:
- GOAL-move-ctrl-t06-s00: No valid relocation for plate_1
