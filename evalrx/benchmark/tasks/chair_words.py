"""CHAIR object-word extraction — a verbatim port of ``caption_to_words`` from
OPERA's ``chair.py`` (itself LisaAnne/Hallucination ``utils/chair.py``), so the
per-case PASS/FAIL the harness labels with is the SAME rule the official
``chair.py`` scores captions with: nltk word_tokenize -> pos_tag -> WordNet
lemmatize, COCO double-word / baby-animal / passenger-vehicle merges, the
``toilet seat`` rule, then the 80-category synonym table -> node (canonical)
words. The synonym table and double-word list below are copied from that file.

Needs the nltk corpora ``punkt_tab``, ``averaged_perceptron_tagger_eng`` and
``wordnet``: found on nltk's default search path (the bench image installs them
under /usr/share/nltk_data) or under ``$CHAIR_NLTK_DATA``.
"""

from __future__ import annotations

import os
from functools import lru_cache

SYNONYMS_TXT = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''

COCO_DOUBLE_WORDS = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]


@lru_cache(maxsize=1)
def _tables():
    synonyms = [s.strip().split(", ") for s in SYNONYMS_TXT.splitlines() if s.strip()]
    objects, inverse = [], {}
    for group in synonyms:
        objects.extend(group)
        for s in group:
            inverse[s] = group[0]
    double = {w: w for w in COCO_DOUBLE_WORDS}
    for a in ANIMAL_WORDS:
        double["baby %s" % a] = a
        double["adult %s" % a] = a
    for v in VEHICLE_WORDS:
        double["passenger %s" % v] = v
    double["bow tie"] = "tie"
    double["toilet seat"] = "toilet"
    double["wine glas"] = "wine glass"
    return frozenset(objects), inverse, double


@lru_cache(maxsize=1)
def _nltk():
    import nltk

    extra = os.environ.get("CHAIR_NLTK_DATA")
    if extra and extra not in nltk.data.path:
        nltk.data.path.insert(0, extra)
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    return nltk, wordnet, WordNetLemmatizer()


def _wordnet_pos(tag: str, wordnet):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return None


def caption_to_words(caption: str):
    """``(words, node_words, idxs, all_words)`` exactly as OPERA's ``CHAIR.caption_to_words``."""
    nltk, wordnet, wnl = _nltk()
    objects, inverse, double_dict = _tables()
    tokens = nltk.word_tokenize(str(caption or "").lower())
    lemmas = [wnl.lemmatize(tok, pos=_wordnet_pos(tag, wordnet) or wordnet.NOUN)
              for tok, tag in nltk.pos_tag(tokens)]
    i, merged, idxs = 0, [], []
    while i < len(lemmas):
        idxs.append(i)
        pair = " ".join(lemmas[i:i + 2])
        if pair in double_dict:
            merged.append(double_dict[pair])
            i += 2
        else:
            merged.append(lemmas[i])
            i += 1
    words = merged
    if ("toilet" in words) and ("seat" in words):
        words = [w for w in words if w != "seat"]
    idxs = [idxs[k] for k, w in enumerate(words) if w in objects]
    words = [w for w in words if w in objects]
    node_words = [inverse[w] for w in words]
    return words, node_words, idxs, merged


def chair_case(caption: str, gt_objects) -> dict:
    """Per-caption CHAIR bookkeeping: hallucinated (word, node) pairs, recalled GT nodes, length."""
    gt = set(gt_objects or ())
    words, nodes, _idxs, all_words = caption_to_words(caption)
    hallucinated = [(w, n) for w, n in zip(words, nodes) if n not in gt]
    recalled = sorted({n for n in nodes if n in gt})
    return {"hallucinated": hallucinated, "recalled": recalled, "n_mentioned": len(words),
            "recall": (len(recalled) / len(gt)) if gt else 0.0, "len": len(all_words)}
