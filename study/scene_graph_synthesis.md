# Scene graphs for the reproduced VLA: comparison and open questions

This synthesis compares representation choices rather than treating every graph as
the same kind of robot memory. The [matrix](literature_matrix.md) records primary
sources and the inspected scope. Several additions were checked at abstract/project
level only; their detailed experimental limitations are not claimed as reviewed.
These are agent-prepared notes, not confirmation that the student has read each paper.

## Representation: what a node and an edge mean

In image scene graphs, an object node can be grounded to a box and connected to
another node by a predicate. Johnson uses such structure to express image queries;
Xu predicts object and relation labels jointly. Their shared notation is useful
for specifying an instruction target, but does not itself give the node a persistent
physical identity. A node in one image need not be the same track in the next.
[Johnson](https://openaccess.thecvf.com/content_cvpr_2015/html/Johnson_Image_Retrieval_Using_2015_CVPR_paper.html),
[Xu](https://arxiv.org/abs/1701.02426).

Armeni instead organizes a metric environment through building, room, object and
camera entities. Wald predicts semantic relations from reconstructed 3D instances.
The former makes spatial context and observation geometry explicit; the latter
emphasizes learned semantic graph prediction. Neither choice forces a policy to
consume the representation. For tabletop target selection, our inference is that
object instances and their relevant relations may be sufficient; a building hierarchy
would need a separate justification.
[Armeni](https://openaccess.thecvf.com/content_ICCV_2019/html/Armeni_3D_Scene_Graph_A_Structure_for_Unified_Semantics_3D_Space_ICCV_2019_paper.html),
[Wald](https://openaccess.thecvf.com/content_CVPR_2020/html/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.html).

## Construction: prediction, geometry and observation coverage

Xu and MotifNet learn predicates from annotated images and contextual statistics.
ConceptGraphs associates observations across posed RGB-D frames into 3D objects,
with foundation-model descriptors and graph relations. These pipelines have different
failure surfaces: a frequent category-pair predicate may be wrong for the particular
image, whereas a fused object may have an incorrect association. Correct graph
syntax is therefore insufficient evidence of correct grounding.
[Xu](https://arxiv.org/abs/1701.02426),
[MotifNet](https://arxiv.org/abs/1711.06640),
[ConceptGraphs](https://concept-graphs.github.io/).

For our proposed spatial edges, metric geometry permits an explicit operational
definition of relations such as proximity. That improves auditability, but creates
threshold and reference-frame choices. It does not establish that 3D is universally
better than 2D. In particular, the S2 question requires checking whether the moved
landmark is sufficiently visible in both policy inputs before attributing a wrong
choice specifically to relation computation. An oracle simulator graph can supply
information unavailable in the images; a gain under that interface must be labelled
as privileged-information evidence.

## Time: incremental maps, identity and events

Action Genome expresses changing human-object relations in video. Rosinol's dynamic
scene graphs combine spatial hierarchy with moving agents, while Hydra constructs
and optimizes layered spatial graphs online. These are different meanings of
temporal structure: a sequence of predicates, dynamic entity trajectories, and an
incrementally corrected map. None of those labels alone proves that visually
identical objects retain correct identities through a swap.
[Action Genome](https://arxiv.org/abs/1912.06992),
[Dynamic scene graphs](https://www.roboticsproceedings.org/rss16/p079.html),
[Hydra](https://www.roboticsproceedings.org/rss18/p050.html).

Our temporal instruction needs the relation that held at an earlier time and the
identity of the object that held it. A useful state would therefore distinguish
current pose from a historical event, track identity confidence, and record whether
a change was observed or merely assumed. Memory cannot reveal an unobserved motion
with certainty. The historical T1 rollouts also diverged before first placement;
they motivate this interface discussion without validating the predeclared temporal
bound on those recorded pairs.

## Robot use: planning and policy conditioning

SayPlan uses task-relevant hierarchical subgraphs and feedback to support high-level
planning. RoboEXP adds action-conditioned structure through interactive exploration
and explicit memory. The former emphasizes selecting and checking action sequences;
the latter emphasizes discovering structure through interaction. Both make a graph
operational, but neither is the current pi0 interface of two RGB images, robot state
and a prompt.
[SayPlan](https://proceedings.mlr.press/v229/rana23a.html),
[RoboEXP](https://proceedings.mlr.press/v270/jiang25c.html).

Directly related graph/VLA work also exists. GraSP-VLA describes graph-derived
symbolic orchestration of low-level VLA policies. GF-VLA describes temporal interaction
graphs and hierarchical dual-arm execution. SceneDiver describes graph-supported
focus planning with an adapter for VLA distillation. These verified primary abstracts
already rule out a blanket novelty claim that no previous system combines graphs
and VLAs. Their exact interfaces, training requirements and evaluation controls
need full-text comparison before claiming a narrower methodological gap.
[GraSP-VLA](https://arxiv.org/abs/2511.04357),
[GF-VLA v2](https://arxiv.org/abs/2508.05342v2),
[SceneDiver](https://arxiv.org/abs/2606.04046).

For this assignment, the least invasive candidate is a resolver that maps a relation
query to an instance and rewrites it into an instruction the fixed policy can follow.
S1 makes this plausible, not guaranteed under moved layouts. Alternative interfaces
include a target mask/visual prompt or learned graph tokens. Each changes the policy's
input distribution differently and needs its own control. A successful oracle
rewrite would not establish perception robustness or the necessity of graph tokens.

## Evaluation and limits

Image-graph prediction and retrieval in Johnson/Xu differ from 3D graph prediction
and retrieval in Wald. Those metrics test representational tasks, whereas SayPlan
and RoboEXP use the representation in robot systems. Our downstream metric must
remain identity-specific task success, accompanied by wrong-object choices and
execution latency. A graph can have many correct edges yet fail on the single edge
that determines the target.
[Johnson](https://openaccess.thecvf.com/content_cvpr_2015/html/Johnson_Image_Retrieval_Using_2015_CVPR_paper.html),
[Xu](https://arxiv.org/abs/1701.02426),
[Wald](https://openaccess.thecvf.com/content_CVPR_2020/html/Wald_Learning_3D_Semantic_Scene_Graphs_From_3D_Indoor_Reconstructions_CVPR_2020_paper.html),
[SayPlan](https://proceedings.mlr.press/v229/rana23a.html),
[RoboEXP](https://proceedings.mlr.press/v270/jiang25c.html).

Our proposed controls separate a fixed policy, an oracle target resolver, and any
perception-derived resolver. The same eligible episodes, instruction templates and
noise contract are used for paired conditions. Full task predicates and first placed
identity answer different questions. Failure labels must distinguish a wrong target
from a correct target that was not successfully placed. Sample size and task coverage
must accompany rates; several states of one task are not independent new task types.

## Concrete research questions

These are questions for our design and evaluation, not claims that prior work has
never addressed them.

1. Which minimum node attributes and relations resolve an identical-object instruction
   without carrying irrelevant scene structure?
2. How should a resolver express uncertainty when a landmark is clipped, occluded or
   missing from both cameras?
3. Which geometric relation thresholds agree with the task's intended semantics, and
   how sensitive is target selection to those thresholds?
4. What evidence preserves identity through a swap, and when should tracking abstain
   instead of assigning a confident historical identity?
5. How should graph corrections invalidate stale relations and instruction rewrites
   after object motion or map optimization?
6. Does graph-resolved language improve target selection beyond a wording-only control,
   using a fixed checkpoint and the same episodes?
7. If target selection improves but placement still fails, what information would a
   continuous-action interface actually need from the graph?
8. How much of an oracle-graph gain survives when the graph is built only from the
   policy's available observations?

## Implications for this assignment

The evidence motivates testing explicit, updated instance relations before a large
perception or fine-tuning stack. Preserve the fixed pi0 baseline and diagnose whether
the error is visible target grounding, unavailable history, or low-level execution.
Use oracle graphs only as a declared upper-bound interface experiment, then introduce
perception errors separately. The contribution can be a reproducible diagnosis and
a falsifiable improvement proposal; it need not claim a new scene-graph paradigm.

Search record (2026-10-30): primary proceedings/project pages for 3D scene graphs,
3DSSG, dynamic scene graphs, Hydra, ConceptGraphs, SayPlan and RoboEXP; targeted queries
for `scene graph` with `vision-language-action` and `VLA policy`; verified arXiv entries
for GraSP-VLA, GF-VLA and SceneDiver. This is a bounded search, not an exhaustive review.
