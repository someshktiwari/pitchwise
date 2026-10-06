"""
evaluation/build_tests_v3.py

Builds evaluation/tests_v3.jsonl, the test set for the expanded knowledge
base (DECISIONS.md D-014), from the frozen v2 set (evaluation/tests.jsonl):

- v2 questions whose answers changed are relabelled or rewritten, each with
  its reason below (out-of-scope questions the Wikipedia articles now answer,
  Ben Stokes's 2026 retirement, C-007);
- new questions cover the Wikipedia documents;
- every keyword of every answerable question is checked against the document
  it should come from, so no question asks about text the knowledge base
  doesn't contain (the C-004 lesson).

Run from the project root:  uv run python -m evaluation.build_tests_v3
Question numbers below are line numbers in evaluation/tests.jsonl.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge-base"
v2 = [json.loads(l) for l in (ROOT / "evaluation/tests.jsonl").read_text().splitlines() if l.strip()]
assert len(v2) == 104

def q(question, keywords, reference, category, doc=None):
    return {"question": question, "keywords": keywords, "reference_answer": reference,
            "category": category, **({"_doc": doc} if doc else {})}

# ---- changes to v2 questions (1-based numbers) ----
changes = {
    21: q("Between which years was Ben Stokes England's Test captain?", ["Stokes", "April 2022", "2026"],
          "Ben Stokes was England's Test captain from April 2022 until he retired from international cricket in 2026.",
          "temporal", "players/ben-stokes.md"),
    78: q("How many overs separate the powerplay lengths of ODI and T20I cricket?", ["first 10 overs", "6 overs"],
          "The ODI powerplay covers the first 10 overs and the T20I powerplay the first 6 overs, so they differ by 4 overs.",
          "spanning"),
    83: q("Has Ben Stokes retired from any international format?", ["Stokes", "retired", "2026"],
          "Yes. Ben Stokes retired from ODI cricket after the 2023 World Cup, from T20Is in 2025, and from international cricket altogether in 2026; his last Test was against New Zealand in June 2026.",
          "temporal", "players/ben-stokes.md"),
    88: q("Who succeeded Ben Stokes as England's Test captain?", ["Joe Root", "reappointed"],
          "Joe Root, who was reappointed England's Test captain in 2026 after Ben Stokes retired; Root had previously captained the Test team from 2017 to 2022.",
          "temporal"),
    90: q("Which batter is known as the 'Chase Master' for his record in ODI run chases?", ["Chase Master", "Kohli"],
          "Virat Kohli is known as the 'Chase Master' for his exceptional record batting second in ODI run chases.",
          "direct_fact", "players/virat-kohli.md"),
    91: q("Which three ICC trophies did MS Dhoni win as India's captain?", ["Dhoni", "2007", "2011", "2013"],
          "MS Dhoni led India to the 2007 ICC World Twenty20, the 2011 Cricket World Cup and the 2013 ICC Champions Trophy, the only captain to win all three ICC white-ball trophies.",
          "direct_fact", "wikipedia/players/ms-dhoni.md"),
    92: q(None, ["Champions Trophy", "2025", "India"],
          "India won the 2025 ICC Champions Trophy, their third title in the tournament (after 2002 and 2013).",
          "direct_fact", "wikipedia/tournaments/icc-champions-trophy.md"),
    99: q(None, ["resources", "overs", "wickets"],
          "The DLS method treats a batting side as having two resources, overs remaining and wickets in hand, and adjusts the target in proportion to the resources lost when overs are cut.",
          "direct_fact", "wikipedia/game/duckworth-lewis-stern-method.md"),
    101: q(None, ["Big Bash", "eight", "city-based"],
           "The Big Bash League has eight city-based franchise teams; it replaced a competition of six state-based teams.",
           "direct_fact", "wikipedia/tournaments/big-bash-league.md"),
    102: q(None, ["Muralitharan", "800"],
           "Muttiah Muralitharan took 800 Test wickets, the only bowler to reach 800.",
           "direct_fact", "wikipedia/players/muttiah-muralitharan.md"),
    # C-009: answerable after all; found by the v3 linear run, where the model
    # answered 40.57 correctly from the context and the old label scored it 1
    95: q(None, ["Rohit", "4,301", "40.57"],
          "Rohit Sharma finished his Test career with 4,301 runs in 67 Tests at an average of 40.57.",
          "direct_fact", "wikipedia/players/rohit-sharma.md"),
    104: q(None, ["Mohammad Ashraful", "youngest"],
           "Mohammad Ashraful of Bangladesh became the youngest player to score a Test century, doing so in his first match.",
           "direct_fact", "wikipedia/teams/bangladesh-national-cricket-team.md"),
}
oos_refs = {
    94: "The knowledge base does not say who India's current head coach is (it mentions Rahul Dravid replacing Ravi Shastri in 2021, not who coaches the team now).",
    96: "There was no 2025 Women's T20 World Cup in the knowledge base: it lists New Zealand as 2024 champions and Australia as 2026 champions.",
}
KEYWORD_FIXES = {15: ["Tendulkar", "15,921"]}  # v2 also listed "15921", which no document contains
DROP = {98}  # fastest T20I century: the only claim is a 2017 'joint-fastest' record, no longer true

out = []
for n, t in enumerate(v2, 1):
    if n in DROP:
        continue
    t = dict(t)
    if n in changes:
        c = changes[n]
        for k in ("keywords", "reference_answer", "category"):
            t[k] = c[k]
        if c["question"]:
            t["question"] = c["question"]
        if "_doc" in c:
            t["_doc"] = c["_doc"]
    if n in KEYWORD_FIXES:
        t["keywords"] = KEYWORD_FIXES[n]
    if n in oos_refs:
        t["reference_answer"] = oos_refs[n]
    out.append(t)

W = "wikipedia/"
new = [
    # ---- direct facts from the new documents ----
    q("What is Brian Lara's highest score in a Test innings?", ["Lara", "400 not out"], "400 not out, for the West Indies against England at Antigua in 2004, the highest individual score in Test cricket.", "direct_fact", W+"players/brian-lara.md"),
    q("What is the highest individual score in first-class cricket, and who made it?", ["501 not out", "Durham"], "Brian Lara's 501 not out for Warwickshire against Durham at Edgbaston in 1994.", "direct_fact", W+"players/brian-lara.md"),
    q("What is the capacity of Eden Gardens?", ["Eden Gardens", "68,000"], "Eden Gardens in Kolkata has a capacity of 68,000.", "direct_fact", W+"grounds/eden-gardens.md"),
    q("What is the capacity of the Narendra Modi Stadium?", ["Narendra Modi Stadium", "132,000"], "132,000, making it the world's largest cricket stadium.", "direct_fact", W+"grounds/narendra-modi-stadium.md"),
    q("Who owns Lord's Cricket Ground, and who is it named after?", ["Marylebone Cricket Club", "Thomas Lord"], "Lord's is owned by Marylebone Cricket Club (MCC) and is named after its founder, Thomas Lord.", "direct_fact", W+"grounds/lord-s.md"),
    q("Which was the first ground in England to host Test cricket?", ["The Oval", "1880"], "The Oval, which hosted England's first Test in September 1880.", "direct_fact", W+"grounds/the-oval.md"),
    q("Who is Lahore's Gaddafi Stadium named after?", ["Gaddafi", "Muammar Gaddafi"], "It is named after the Libyan revolutionary Muammar Gaddafi.", "direct_fact", W+"grounds/gaddafi-stadium.md"),
    q("Since when has Headingley hosted Test cricket, and what is its capacity?", ["Headingley", "1899", "18,350"], "Headingley has hosted Test cricket since 1899 and has a capacity of 18,350.", "direct_fact", W+"grounds/headingley-cricket-ground.md"),
    q("How many Test wickets did Anil Kumble take?", ["Kumble", "619"], "Anil Kumble took 619 Test wickets.", "direct_fact", W+"players/anil-kumble.md"),
    q("How many Test wickets did Curtly Ambrose take, and at what average?", ["Ambrose", "405", "20.99"], "Curtly Ambrose took 405 Test wickets at an average of 20.99.", "direct_fact", W+"players/curtly-ambrose.md"),
    q("What is notable about Malcolm Marshall's Test bowling average?", ["Marshall", "20.94"], "Malcolm Marshall's Test bowling average of 20.94 is the best of any bowler with 300 or more Test wickets.", "direct_fact", W+"players/malcolm-marshall.md"),
    q("How many first-class runs and centuries did Jack Hobbs score?", ["Hobbs", "61,760", "199"], "Jack Hobbs scored 61,760 first-class runs and 199 centuries, the most of anyone.", "direct_fact", W+"players/jack-hobbs.md"),
    q("What unique Test double does Kapil Dev hold?", ["Kapil Dev", "434", "5,000"], "Kapil Dev is the only player to take more than 400 Test wickets (434) and score more than 5,000 Test runs.", "direct_fact", W+"players/kapil-dev.md"),
    q("Which World Cup did Imran Khan win as Pakistan's captain?", ["Imran Khan", "1992"], "Imran Khan captained Pakistan to victory in the 1992 Cricket World Cup.", "direct_fact", W+"players/imran-khan.md"),
    q("Who was the first batsman to pass 10,000 runs in Test cricket?", ["Gavaskar", "10,000"], "Sunil Gavaskar.", "direct_fact", W+"players/sunil-gavaskar.md"),
    q("What nicknames is Rahul Dravid known by?", ["Dravid", "The Wall", "Mr. Dependable"], "Rahul Dravid is known as 'The Wall' and 'Mr. Dependable'.", "direct_fact", W+"players/rahul-dravid.md"),
    q("What is Ricky Ponting's record as an international captain?", ["Ponting", "220", "324"], "Ricky Ponting is the most successful captain in international cricket, with 220 wins in 324 matches (67.91%).", "direct_fact", W+"players/ricky-ponting.md"),
    q("Which World Cup did Allan Border win as Australia's captain?", ["Border", "1987"], "Allan Border led Australia to victory in the 1987 Cricket World Cup, Australia's first world title.", "direct_fact", W+"players/allan-border.md"),
    # C-009: was "the first player to score a double century in ODI cricket?", which the
    # knowledge base answers two ways (Belinda Clark in 1997; the curated Tendulkar
    # document says Sachin, true only of men's ODIs)
    q("Who was the first woman to score a double century in a One Day International?", ["Belinda Clark", "double century"], "Belinda Clark of Australia, the first player to record a double century in the ODI format.", "direct_fact", W+"players/belinda-clark.md"),
    q("How many ODI wickets did Jhulan Goswami take?", ["Goswami", "255"], "Jhulan Goswami took 255 wickets in 204 ODIs, the most in women's ODI cricket.", "direct_fact", W+"players/jhulan-goswami.md"),
    q("Who captained England when they won the first Women's Cricket World Cup?", ["Heyhoe Flint", "1973"], "Rachael Heyhoe Flint captained England to victory in the inaugural 1973 Women's Cricket World Cup.", "direct_fact", W+"players/rachael-heyhoe-flint.md"),
    q("What made Ellyse Perry's international debut unusual?", ["Perry", "16", "FIFA"], "Ellyse Perry debuted for both Australia's national cricket and soccer teams at 16, and was the first Australian to play in both ICC and FIFA World Cups.", "direct_fact", W+"players/ellyse-perry.md"),
    q("Who captained India to the 2025 Women's Cricket World Cup title?", ["Harmanpreet", "2025 Women's Cricket World Cup"], "Harmanpreet Kaur.", "direct_fact", W+"players/harmanpreet-kaur.md"),
    q("What was Steve Smith's peak ICC Test batting rating, and how does it compare with Don Bradman's?", ["947", "961"], "Steve Smith reached an ICC Test batting rating of 947, the second-highest of all time behind Don Bradman's 961.", "direct_fact", W+"players/steve-smith-cricketer.md"),
    q("Which award did Shaheen Shah Afridi win in 2021?", ["Afridi", "2021", "Sir Garfield Sobers Trophy"], "He was named ICC Men's Cricketer of the Year 2021, the first Pakistani to win the Sir Garfield Sobers Trophy.", "direct_fact", W+"players/shaheen-shah-afridi.md"),
    q("Who became the most expensive player in IPL history, and for how much?", ["Pant", "27.00 crore", "Lucknow Super Giants"], "Rishabh Pant, bought by Lucknow Super Giants for ₹27 crore in the 2025 auction.", "direct_fact", W+"players/rishabh-pant.md"),
    q("Which three consecutive World Cups did Glenn McGrath win with Australia?", ["McGrath", "1999", "2003", "2007"], "The 1999, 2003 and 2007 Cricket World Cups.", "direct_fact", W+"players/glenn-mcgrath.md"),
    q("How many Test and ODI wickets did Courtney Walsh take?", ["Walsh", "519", "227"], "Courtney Walsh took 519 Test wickets and 227 ODI wickets.", "direct_fact", W+"players/courtney-walsh.md"),
    q("Which team is the most successful in the Big Bash League?", ["Perth Scorchers"], "The Perth Scorchers.", "direct_fact", W+"tournaments/big-bash-league.md"),
    q("How many teams play in the SA20, and which team has won it most often?", ["SA20", "six teams", "Sunrisers Eastern Cape"], "Six teams; Sunrisers Eastern Cape have won three of the first four editions.", "direct_fact", W+"tournaments/sa20.md"),
    q("What format does The Hundred use, and how many teams take part?", ["100-ball", "eight teams"], "The Hundred uses a 100-ball format and has eight teams, seven in England and one in Wales.", "direct_fact", W+"tournaments/the-hundred-cricket.md"),
    q("When was the County Championship established, and how many clubs contest it?", ["County Championship", "1890", "18 clubs"], "It was established in 1890 and is contested by 18 clubs.", "direct_fact", W+"tournaments/county-championship.md"),
    q("Who is the Sheffield Shield named after, and when was it first contested?", ["Lord Sheffield", "1892–93"], "It is named after Lord Sheffield and was first contested in the 1892–93 season.", "direct_fact", W+"tournaments/sheffield-shield.md"),
    q("Who is the Frank Worrell Trophy named after, and which teams contest it?", ["Frank Worrell", "first black captain"], "It is named after Frank Worrell, the first black captain of the West Indies, and is awarded to the winner of West Indies–Australia Test series.", "direct_fact", W+"tournaments/frank-worrell-trophy.md"),
    q("Who is the Border–Gavaskar Trophy named after?", ["Allan Border", "Sunil Gavaskar"], "Former captains Allan Border of Australia and Sunil Gavaskar of India; it is contested by India and Australia in Tests.", "direct_fact", W+"tournaments/border-gavaskar-trophy.md"),
    q("How many teams contested the Ranji Trophy in the 2025–26 season?", ["Ranji Trophy", "38 teams"], "38 teams.", "direct_fact", W+"tournaments/ranji-trophy.md"),
    q("Who won the first Caribbean Premier League?", ["Jamaica Tallawahs"], "The Jamaica Tallawahs, who beat the Guyana Amazon Warriors in the final.", "direct_fact", W+"tournaments/caribbean-premier-league.md"),
    q("What was Bodyline, and against whom was it devised?", ["Bodyline", "1932–33", "Bradman"], "Bodyline (fast leg theory) was a tactic devised by England for the 1932–33 Ashes tour, bowling at the batsman's body to combat Don Bradman.", "direct_fact", W+"game/bodyline.md"),
    q("How does a Super Over work?", ["Super Over", "six balls"], "If a limited-overs match is tied, each team bats one extra over of six balls, and the team scoring more runs wins.", "direct_fact", W+"game/super-over.md"),
    q("Which Law of Cricket governs the wicket-keeper?", ["wicket-keeper", "Law 27"], "Law 27.", "direct_fact", W+"game/wicket-keeper.md"),
    q("Where is the BCCI headquartered, and who was its first president?", ["Churchgate", "Grant Govan"], "At the Cricket Centre in Churchgate, Mumbai; R. E. Grant Govan was its first president.", "direct_fact", W+"game/board-of-control-for-cricket-in-india.md"),
    q("When was the Marylebone Cricket Club founded, and when was it cricket's governing body?", ["1787", "1788", "1909"], "MCC was founded in 1787 and served as cricket's governing body from 1788 to 1909.", "direct_fact", W+"game/marylebone-cricket-club.md"),
    q("What are the two kinds of review in the Decision Review System?", ["Umpire Review", "Player Review"], "An Umpire Review, where on-field umpires consult the third umpire, and a Player Review, where players ask the third umpire to reconsider a decision.", "direct_fact", W+"game/decision-review-system.md"),
    q("When was the first recorded women's cricket match?", ["26 July 1745"], "On 26 July 1745, in England.", "direct_fact", W+"game/women-s-cricket.md"),
    q("What is the nickname of New Zealand's men's team, and when did they win their first Test?", ["Black Caps", "1956"], "The Black Caps; their first Test win came in 1956, against the West Indies at Eden Park, Auckland.", "direct_fact", W+"teams/new-zealand-national-cricket-team.md"),
    q("Where does the South African team's nickname come from?", ["Proteas", "King Protea"], "The Proteas are named after South Africa's national flower, Protea cynaroides, the King Protea.", "direct_fact", W+"teams/south-africa-national-cricket-team.md"),
    q("Since when has Zimbabwe been a Full Member of the ICC, and what is the team's nickname?", ["Zimbabwe", "1992", "Chevrons"], "Since 1992; the team is known as the Chevrons.", "direct_fact", W+"teams/zimbabwe-national-cricket-team.md"),
    q("When was the Under-19 Cricket World Cup first held?", ["1988", "Youth Cricket World Cup"], "In 1988, as the Youth Cricket World Cup.", "direct_fact", W+"tournaments/under-19-men-s-cricket-world-cup.md"),

    # ---- spanning: facts from two documents ----
    q("How much larger is the Narendra Modi Stadium's capacity than Eden Gardens'?", ["132,000", "68,000"], "The Narendra Modi Stadium holds 132,000 and Eden Gardens 68,000, a difference of 64,000.", "spanning"),
    q("Compare the Test wicket tallies of Anil Kumble and Muttiah Muralitharan.", ["619", "800"], "Muralitharan took 800 Test wickets and Kumble 619, so Muralitharan took 181 more.", "spanning"),
    q("Which captains won the 1983 and 1992 Cricket World Cups?", ["Kapil Dev", "Imran Khan"], "Kapil Dev captained India to the 1983 title and Imran Khan captained Pakistan to the 1992 title.", "spanning"),
    q("Who took more Test wickets, Curtly Ambrose or Courtney Walsh, and by how many?", ["405", "519"], "Courtney Walsh took 519 Test wickets and Curtly Ambrose 405, so Walsh took 114 more.", "spanning"),
    q("By how many runs does Brian Lara's highest Test score exceed Matthew Hayden's?", ["400", "380"], "Lara's 400 not out exceeds Hayden's 380 by 20 runs.", "spanning"),
    q("Which grounds are home to the Sydney Sixers and the Brisbane Heat?", ["Sydney Sixers", "Brisbane Heat"], "The Sydney Cricket Ground is home to the Sydney Sixers and the Gabba to the Brisbane Heat.", "spanning"),
    q("Compare the number of teams in the Big Bash League and the SA20.", ["eight", "six"], "The Big Bash League has eight city-based franchises and the SA20 has six teams.", "spanning"),
    q("Who captained India and who was head coach when they won the 2024 T20 World Cup?", ["Rohit", "Dravid", "2024"], "Rohit Sharma captained India to the 2024 T20 World Cup title, with Rahul Dravid as head coach.", "spanning"),
    q("Who are the three highest century-makers in international cricket?", ["Tendulkar", "Kohli", "Ponting"], "Sachin Tendulkar, Virat Kohli and Ricky Ponting, in that order.", "spanning"),
    q("Which competition is older, England's County Championship or Australia's Sheffield Shield?", ["1890", "1892–93"], "The County Championship (established 1890) is older than the Sheffield Shield (first contested 1892–93).", "spanning"),
    q("Who captained India and who was vice-captain when they won the 2025 Women's Cricket World Cup?", ["Harmanpreet", "Mandhana"], "Harmanpreet Kaur captained the side; Smriti Mandhana, India's vice-captain, was also part of the winning team.", "spanning"),
    q("Which Test records do Mohammad Ashraful and Muttiah Muralitharan hold?", ["Ashraful", "youngest", "800"], "Mohammad Ashraful is the youngest player to score a Test century, and Muttiah Muralitharan is the only bowler with 800 Test wickets.", "spanning"),

    # ---- temporal ----
    q("Who are the current ICC World Test Champions?", ["South Africa", "2025"], "South Africa, who beat Australia in the 2025 final at Lord's.", "temporal", W+"tournaments/world-test-championship.md"),
    q("Is Brendon McCullum still England's Test coach?", ["McCullum", "July 2026"], "No. He was sacked as England's Test coach in July 2026, though he remains head coach of England's T20 and ODI sides.", "temporal", W+"players/brendon-mccullum.md"),
    q("Who captains India in Test cricket as of 2026?", ["Gill", "captains India in Tests"], "Shubman Gill captains India in Tests and ODIs.", "temporal", W+"players/shubman-gill.md"),
    q("Which team are the current Women's T20 World Cup champions?", ["Australia", "2026"], "Australia, who won the 2026 edition for a record seventh title.", "temporal", W+"tournaments/women-s-t20-world-cup.md"),
    q("Is Garfield Sobers still alive?", ["Sobers", "17 July 2026"], "No. Sir Garfield Sobers died on 17 July 2026.", "temporal", W+"players/garfield-sobers.md"),

    # ---- out of scope: the subject may be mentioned, the fact is not ----
    q("How many Test wickets has Nathan Lyon taken in his career?", [], "The knowledge base mentions Nathan Lyon but does not give his career Test wicket tally.", "out_of_scope"),
    q("What is Jos Buttler's highest ODI score?", [], "The knowledge base does not contain Jos Buttler's highest ODI score.", "out_of_scope"),
    q("Which team won the 2023 Major League Cricket final?", [], "The knowledge base mentions Major League Cricket but not who won the 2023 final.", "out_of_scope"),
    q("What is Quinton de Kock's ODI batting average?", [], "The knowledge base does not contain Quinton de Kock's ODI batting average.", "out_of_scope"),
    q("What is the seating capacity of Pallekele International Cricket Stadium?", [], "The knowledge base mentions Pallekele but does not give its capacity.", "out_of_scope"),
    q("Who won the 2024 T20 Blast in England?", [], "The knowledge base mentions the T20 Blast but not the 2024 winner.", "out_of_scope"),
    q("How many ODI wickets did Lasith Malinga take?", [], "The knowledge base mentions Lasith Malinga but not his ODI wicket tally.", "out_of_scope"),
    q("Who is the chief executive of Cricket Australia?", [], "The knowledge base does not say who Cricket Australia's chief executive is.", "out_of_scope"),
    q("What is the capacity of the HPCA Stadium in Dharamsala?", [], "The knowledge base does not cover the HPCA Stadium in Dharamsala.", "out_of_scope"),
]
out += new

# ---- checks ----
texts = {str(p.relative_to(KB)): p.read_text(encoding="utf-8") for p in KB.rglob("*.md")}
all_text = "\n".join(texts.values()).lower()
problems = []
qs = [t["question"] for t in out]
if len(qs) != len(set(qs)):
    problems.append("duplicate questions")
for t in out:
    doc = t.pop("_doc", None)
    if t["category"] == "out_of_scope":
        if t["keywords"]:
            problems.append(f"OOS with keywords: {t['question']}")
        continue
    src = texts[doc].lower() if doc else all_text
    for k in t["keywords"]:
        if k.lower() not in src:
            problems.append(f"missing {k!r} in {doc or 'KB'}: {t['question']}")
from collections import Counter
print(len(out), Counter(t["category"] for t in out))
print("\n".join(problems) or "all keywords found")
(ROOT / "evaluation/tests_v3.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in out) + "\n")
