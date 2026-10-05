---
title: Duckworth–Lewis–Stern method
source: https://en.wikipedia.org/w/index.php?title=Duckworth%E2%80%93Lewis%E2%80%93Stern_method&oldid=1375295034
revision_timestamp: 2026-09-17T00:00:35Z
retrieved: 2026-10-05
licence: CC BY-SA 4.0, Wikipedia contributors. Adapted: reformatted into markdown and trimmed.
---
# Duckworth–Lewis–Stern method

## Duckworth–Lewis–Stern method: Overview

The Duckworth–Lewis–Stern method (DLS method or DLS) previously known as the Duckworth–Lewis method (D/L) is a mathematical formulation designed to calculate the target score (number of runs needed to win) for the team batting second in a limited overs cricket match interrupted by weather or other circumstances. The method was devised by two English statisticians, Frank Duckworth and Tony Lewis, and was formerly known as the Duckworth–Lewis method (D/L). It was introduced in 1997, and adopted officially by the International Cricket Council (ICC) in 1999. After the retirement of both Duckworth and Lewis, the Australian statistician Steven Stern became the custodian of the method, which was renamed to its current title in November 2014. In 2014, he refined the model to better fit modern scoring trends, especially in T20 cricket, resulting in the updated Duckworth-Lewis-Stern method. This refined method remains the standard for handling rain-affected matches in international cricket today.
The target score in cricket matches without interruptions is one more than the number of runs scored by the team that batted first. When overs are lost, setting an adjusted target for the team batting second is not as simple as reducing the run target proportionally to the loss in overs, because a team with ten wickets in hand and 25 overs to bat can play more aggressively than if they had ten wickets and a full 50 overs, for example, and can consequently achieve a higher run rate. The DLS method is an attempt to set a statistically fair target for the second team's innings, which is the same difficulty as the original target. The basic principle is that each team in a limited-overs match has two resources available with which to score runs (overs to play and wickets remaining), and the target is adjusted proportionally to the change in the combination of these two resources.

## Duckworth–Lewis–Stern method: History and creation

Various different methods had been used previously to resolve rain-affected cricket matches, with the most common being the Average Run Rate method, and later, the Most Productive Overs method.
While simple in nature, these methods had intrinsic flaws and were easily exploitable:

The Average Run Rate method took no account of wickets lost by the team batting second, but simply reflected their scoring rate when the match was interrupted. If the team felt a rain stoppage was likely, they could attempt to force the scoring rate with no regard for the corresponding highly likely loss of wickets, meaning any comparison with the team batting first would be flawed.
The Most Productive Overs method not only took no account of wickets lost by the team batting second, but also effectively penalised the team batting second for good bowling by ignoring their best overs in setting the revised target.
Both of these methods also produced revised targets that frequently altered the balance of the match, and they took no account of the match situation at the time of the interruption.
The D/L method was devised by two British statisticians, Frank Duckworth and Tony Lewis, as a result of the outcome of the semi-final in the 1992 World Cup between England and South Africa, where the Most Productive Overs method was used. When rain stopped play for 12 minutes, South Africa needed 22 runs from 13 balls, but when play resumed, the revised target left South Africa needing 21 runs from one ball, a reduction of only one run compared to a reduction of two overs, and a virtually impossible target given that the maximum score from one ball is generally six runs. Duckworth said, "I recall hearing Christopher Martin-Jenkins on radio saying 'surely someone, somewhere could come up with something better' and I soon realised that it was a mathematical problem that required a mathematical solution." The D/L method avoids this flaw: in this match, the revised D/L target of 236 would have left South Africa needing four to tie or five to win from the final ball, assuming that South Africa's tactic of deliberately delaying their bowling innings to reduce the match length wasn't punished in some fashion.
The D/L method was first used in international cricket on 1 January 1997 in the second match of the Zimbabwe versus England ODI series, which Zimbabwe won by seven runs. The D/L method was formally adopted by the ICC in 1999 as the standard method of calculating target scores in rain-shortened one-day matches.

## Duckworth–Lewis–Stern method: Theory - Calculation summary

The essence of the D/L method is 'resources'. Each team is taken to have two 'resources' to use to score as many runs as possible: the number of overs they have to receive; and the number of wickets they have in hand. At any point in any innings, a team's ability to score more runs depends on the combination of these two resources they have left. Looking at historical scores, there is a very close correspondence between the availability of these resources and a team's final score, a correspondence which D/L exploits.
The D/L method converts all possible combinations of overs (or, more accurately, balls) and wickets left into a combined resources remaining percentage figure (with 50 overs and 10 wickets = 100%), and these are all stored in a published table or computer. The target score for the team batting second ('Team 2') can be adjusted up or down from the total the team batting first ('Team 1') achieved using these resource percentages, to reflect the loss of resources to one or both teams when a match is shortened one or more times.
In the version of D/L most commonly in use in international and first-class matches (the 'Professional Edition'), the target for Team 2 is adjusted simply in proportion to the two teams' resources, i.e.

Team 2's par score

=

Team 1's score

×

Team 2's resources
Team 1's resources

.

If, as usually occurs, this 'par score' is a non-integer number of runs, then Team 2's target to win is this number rounded up to the next integer, and the score to tie (also called the par score), is this number rounded down to the preceding integer. If Team 2 reaches or passes the target score, then they have won the match. If the match ends when Team 2 has exactly met (but not passed) the par score then the match is a tie. If Team 2 fail to reach the par score then they have lost.
For example, if a rain delay means that Team 2 only has 90% of resources available, and Team 1 scored 254 with 100% of resources available, then 254 × 90% / 100% = 228.6, so Team 2's target is 229, and the score to tie is 228. The actual resource values used in the Professional Edition are not publicly available, so a computer which has this software loaded must be used.
If it is a 50-over match and Team 1 completed its innings uninterrupted, then they had 100% resource available to them, so the formula simplifies to:

Team 2's par score

=

Team 1's score

×

Team 2's resources

.

## Duckworth–Lewis–Stern method: Theory - Summary of impact on Team 2's target

If there is a delay before the first innings starts, so that the numbers of overs in the two innings are reduced but still the same as each other, then D/L makes no change to the target score, because both sides are aware of the total number of overs and wickets throughout their innings, thus they will have the same resources available.
Team 2's target score is first calculated once Team 1's innings has finished.
If there were interruption(s) during Team 1's innings, or Team 1's innings was cut short, so the numbers of overs in the two innings are reduced (but still the same as each other), then D/L will adjust Team 2's target score as described above. The adjustment to Team 2's target after interruptions in Team 1's innings is often an increase, implying that Team 2 has more resource available than Team 1 had. Although both teams have 10 wickets and the same (reduced) number of overs available, an increase is fair as, for some of their innings, Team 1 thought they would have more overs available than they actually ended up having. If Team 1 had known that their innings was going to be shorter, they would have batted less conservatively, and scored more runs (at the expense of more wickets). They saved some wicket resource to use up in the overs that ended up being cancelled, which Team 2 does not need to do, therefore Team 2 does have more resource to use in the same number of overs. Therefore, increasing Team 2's target score compensates Team 1 for the denial of some of the overs they thought they would get to bat. The increased target is what D/L thinks Team 1 would have scored in the overs it ended up having, if it had known throughout that the innings would be only as long as it was.
For example, if Team 1 batted for 20 overs before rain came, thinking they would have 50 overs in total, but at the re-start there was only time for Team 2 to bat for 20 overs, it would clearly be unfair to give Team 2 the target that Team 1 achieved, as Team 1 would have batted less conservatively and scored more runs, if they had known they would only have the 20 overs.
If there are interruption(s) to Team 2's innings, either before it starts, during, or it is cut short, then D/L will reduce Team 2's target score from the initial target set at the end of Team 1's innings, in proportion to the reduction in Team 2's resources. If there are multiple interruptions in the second innings, the target will be adjusted downwards each time.
If there are interruptions which both increase and decrease the target score, then the net effect on the target could be either an increase or decrease, depending on whether Team 2's resource loss is large enough.

## Duckworth–Lewis–Stern method: Theory - Mathematical theory

The original D/L model started by assuming that the number of runs that can still be scored (called

Z

), for a given number of overs remaining (called

u

) and wickets lost (called

w

), takes the following exponential decay relationship:

Z
(
u
,
w
)
=

Z

0

(
w
)

(

1
−

e

−
b
(
w
)
u

)

,

where the constant

Z

0

is the asymptotic average total score in unlimited overs (under one-day rules), and

b

is the exponential decay constant. Both vary with

w

(only). The values of these two parameters for each

w

from 0 to 9 were estimated from scores from 'hundreds of one-day internationals' and 'extensive research and experimentation', though were not disclosed due to 'commercial confidentiality'.

Finding the value of

Z

for a particular combination of

u

and

w

(by putting in

u

and the values of these constants for the particular

w

), and dividing this by the score achievable at the start of the innings, i.e. finding

P
(
u
,
w
)
=

Z
(
u
,
w
)

Z
(
u
=
50
,
w
=
0
)

,

gives the proportion of the combined run scoring resources of the innings remaining when

u

overs are left and

w

wickets are down. These proportions can be plotted in a graph, as shown right, or shown in a single table, as shown below.
This became the Standard Edition. When it was introduced, it was necessary that D/L could be implemented with a single table of resource percentages, as it could not be guaranteed that computers would be present. Therefore, this single formula was used giving average resources. This method relies on the assumption that average performance is proportional to the mean, irrespective of the actual score. This was good enough in 95 per cent of matches, but in the 5 per cent of matches with very high scores, the simple approach started to break down. To overcome the problem, an upgraded formula was proposed with an additional parameter whose value depends on the Team 1 innings. This became the Professional Edition.

## Duckworth–Lewis–Stern method: Examples - Stoppage in first innings

**Increased target.**
In the 4th India–England ODI in the 2008 series, the first innings was interrupted by rain on two occasions, reducing the match to 22 overs each. India (batting first) made 166/4. The D/L method increased England's target to 198 from 22 overs. As England knew they had only 22 overs, the expectation is that they could score more runs from those overs than India had from their (interrupted) innings. England made 178/8 from 22 overs, and so the match was listed as "India won by 19 runs (D/L method)".
During the 5th ODI between India and South Africa in January 2011, rain halted play twice during the first innings. The match was reduced to 46 overs each. South Africa scored 250/9. The D/L method increased India's target to 268. As the number of overs was reduced during South Africa's innings, this method takes into account what South Africa were likely to have scored if they had known throughout their innings that it would only be 46 overs long. The match was listed as "South Africa won by 33 runs (D/L method)".

**Decreased target.**
On 3 December 2014, Sri Lanka played England and batted first, but play was interrupted when Sri Lanka had scored 6/1 from 2 overs. At the restart, both innings were reduced to 35 overs, and Sri Lanka finished on 242/8. D/L reduced England's target to 236 from 35 overs. Although Sri Lanka had less resource remaining after the interruption than England would have for their whole innings (about 7% less), they had used up 8% of their resource (2 overs and 1 wicket) before the interruption, so the total resource used by Sri Lanka was still slightly more than England had available, hence the slightly decreased target for England.

## Duckworth–Lewis–Stern method: Examples - Stoppage in second innings

A simple example of the D/L method being applied was the 1st ODI between India and Pakistan in their 2006 ODI series. India batted first, and were all out for 328. Pakistan, batting second, were 311/7 when bad light stopped play after the 47th over. Pakistan's target, had the match continued, was 18 runs in 18 balls, with three wickets in hand. Considering the overall scoring rate throughout the match, this is a target most teams would be favoured to achieve. And indeed, application of the D/L method resulted in a retrospective target score of 305 (or par score of 304) at the end of the 47th over, with the result therefore officially listed as "Pakistan won by 7 runs (D/L method)".
The D/L method was used in the group stage match between Sri Lanka and Zimbabwe at the T20 World Cup in 2010. Sri Lanka scored 173/7 in 20 overs batting first, and in reply Zimbabwe were 4/0 from 1 over when rain interrupted play. At the restart Zimbabwe's target was reduced to 108 from 12 overs, but rain stopped the match when they had scored 29/1 from 5 overs. The retrospective D/L target from 5 overs was a further reduction to 44, or a par score of 43, and hence Sri Lanka won the match by 14 runs.
The DLS method was also used after the rain disruption in the 2023 Indian Premier League final, when Chennai Super Kings had scored 4/0 (0.3 overs) and the Gujarat Titans just scored 214/4 (20 overs). The target was reduced at 171 runs from 15 overs from earlier target of 215 runs from 20 overs for Chennai Super Kings. Chennai Super Kings won by 5 wickets by the DLS method. This was achieved by reaching 171/5 from 15 overs.
An example of a D/L tied match was the ODI between England and India on 11 September 2011. This match was frequently interrupted by rain in the final overs, and a ball-by-ball calculation of the Duckworth–Lewis 'par' score played a key role in tactical decisions during those overs. At one point, India were leading under D/L during one rain delay, and would have won if play had not resumed. At a second rain interval, England, who had scored some quick runs (knowing they needed to get ahead in D/L terms) would correspondingly have won if play had not resumed. Play was finally called off with just 7 balls of the match remaining and England's score equal to the Duckworth–Lewis 'par' score, therefore resulting in a tie.
This example does show how crucial (and difficult) the decisions of the umpires can be, in assessing when rain is heavy enough to justify ceasing play. If the umpires of that match had halted play one ball earlier, England would have been ahead on D/L, and so would have won the match. Equally, if play had stopped one ball later, India could have won the match with a dot ball – indicating how finely-tuned D/L calculations can be in such situations.

## Duckworth–Lewis–Stern method: Examples - Stoppages in both innings

During the 2012–13 Big Bash League season, D/L was used in the 2nd semi-final played between the Melbourne Stars and the Perth Scorchers. After rain delayed the start of the match, it interrupted Melbourne's innings when they had scored 159/1 off 15.2 overs, and both innings were reduced by 2 overs to 18, and Melbourne finished on 183/2. After a further rain delay reduced Perth's innings to 17 overs, Perth returned to the field to face 13 overs, with a revised target of 139. Perth won the game by 8 wickets with a boundary off the final ball.

## Duckworth–Lewis–Stern method: Examples - Stoppage in second innings with revised target already reached

When a team who get their full resources scores a very low total, and their opponents score very quickly early in their innings, a stoppage can result in the calculation of a revised target that has already been reached.
During the 2012–13 Big Bash League season, a match the Perth Scorchers and the Melbourne Stars saw Perth bowled out for a record low total of 69: in response, the Melbourne Stars had scored 29/0 from their first two overs when rain delayed the match.
Once the rain cleared, the umpires decided that the conditions and time remaining was acceptable for a reduced five-over innings from Melbourne, the minimum for a result. Under the older Duckworth-Lewis method, the revised target for their five-over innings was 20, a score that Melbourne had already exceeded: this unusual situation saw the match referee order the two teams to play out a single delivery – a non-scoring leave through to the wicketkeeper – before officially awarding the match to Melbourne, in an effort to avoid confusing the spectators and television viewers.
BBL officials later ruled this delivery was not required, since Melbourne had already won, and the single delivery was deleted from the match result and the calculations of the league table's net run rate. Melbourne won by 24 runs under the D/L method, which was calculated using the par score of five runs after two overs: this par score was incorrectly reported by some media outlets to have been the target.
As any competitive match will have a minimum over requirement—five overs each in Twenty20 and 20 overs each in One Day Internationals. A team being ahead of a revised target for the amount of play remaining can leave the teams waiting for the weather to clear, the ground staff to work and the match referee to decide the game could continue (even though no more play would occur) in order to determine if the match is abandoned or declared a victory for the batting team.
Duckworth and Lewis wrote in 2017 that they had suggested that the calculations involved be done dynamically, and that in these unusual situations, it would mean a team would win the game were they ahead of the par score at any point after overs had begun being lost. They argued that it would also prevent tactics that would otherwise be against the normal spirit of cricket, i.e. scoring runs instead of blocking to get through overs to avoid an abandonment, or a bowling team having their bowlers bowl no-balls or wides in order to prevent a match reaching the minimum requirements.
