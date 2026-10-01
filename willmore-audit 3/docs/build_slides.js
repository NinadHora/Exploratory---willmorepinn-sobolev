// Deck de 10 minutos: problema, métodos, dados, resultados, análise. Simples, sem decoração.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const FONT = "Arial";
const INK = "1A1A1A", MUTED = "555555", HEAD = "1B2A41";
const W = 13.33, M = 0.6, CW = W - 2 * M;

function title(slide, text, sub) {
  slide.addText(text, { x: M, y: 0.35, w: CW, h: 0.8, fontFace: FONT, fontSize: 30, bold: true, color: HEAD, isTextBox: true, margin: 0 });
  if (sub) slide.addText(sub, { x: M, y: 1.1, w: CW, h: 0.45, fontFace: FONT, fontSize: 15, color: MUTED, isTextBox: true, margin: 0 });
}
function qr(slide, file, label, x, y, size = 1.1) {
  slide.addImage({ path: file, x, y, w: size, h: size });
  slide.addText(label, { x: x - 0.4, y: y + size, w: size + 0.8, h: 0.3, fontFace: FONT, fontSize: 9, color: MUTED, align: "center", isTextBox: true, margin: 0 });
}
function para(slide, text, y, h, opts = {}) {
  slide.addText(text, Object.assign({ x: M, y, w: CW, h, fontFace: FONT, fontSize: 16, color: INK, isTextBox: true, margin: 0, valign: "top" }, opts));
}
function bullets(slide, items, y, h, opts = {}) {
  const arr = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(arr, Object.assign({ x: M, y, w: CW, h, fontFace: FONT, fontSize: 16, color: INK, isTextBox: true, margin: 0, valign: "top" }, opts));
}
function table(slide, rows, y, colW, fontSize = 12, opts = {}) {
  const body = rows.map((r, i) => r.map(c => ({ text: c, options: Object.assign({ fontFace: FONT, fontSize, color: INK, valign: "middle", margin: 0.05, border: { type: "solid", color: "BBBBBB", pt: 0.5 } }, i === 0 ? { bold: true, fill: { color: "EEEEEE" } } : {}) })));
  slide.addTable(body, Object.assign({ x: M, y, w: CW, colW, autoPage: false }, opts));
}
function tag(slide, text) {
  slide.addText(text, { x: W - M - 2.2, y: 0.12, w: 2.2, h: 0.3, fontFace: FONT, fontSize: 11, color: MUTED, align: "right", isTextBox: true, margin: 0 });
}

// 1
{
  const s = pres.addSlide();
  s.addText("Can a neural Willmore surface be trusted?", { x: M, y: 2.0, w: CW, h: 1.2, fontFace: FONT, fontSize: 40, bold: true, color: HEAD, isTextBox: true, margin: 0 });
  s.addText("Certificates and adversarial tests from the analysis of the Willmore functional", { x: M, y: 3.2, w: CW, h: 0.6, fontFace: FONT, fontSize: 20, color: MUTED, isTextBox: true, margin: 0 });
  s.addText("Nina da Hora, MSc, Institute of Computing (IC), Unicamp\nMM845 AI for Geometry, mini-project. IMECC, Unicamp, October 2026\nLecturers: Henrique N. Sá Earp, Edward Hirst, Tomás S. R. Silva", { x: M, y: 5.2, w: CW, h: 1.2, fontFace: FONT, fontSize: 14, color: INK, isTextBox: true, margin: 0 });
  s.addNotes("Quando esse paper foi apresentado aqui, a pergunta que ficou foi: como a gente sabe que esse número é verdade? O projeto responde a isso ao pé da letra.");
}

// 2 problema
{
  const s = pres.addSlide(); tag(s, "Problem");
  title(s, "The neural Willmore flow: method and unchecked properties", "Hirst, Sá Earp, Silva. Minimising Willmore energy via neural flow, arXiv:2604.04321 (2026)");
  table(s, [
    ["What the method does", "What is never checked on its output"],
    ["A tanh network phi(u, v) -> R^3 on spectral features; curvatures by automatic differentiation.", "The topology (is the output still a torus?)."],
    ["Loss: Monte Carlo estimate of int H^2 dA on 5000 uniform random points, with a Huber cap on H^2 at 50 and regularity terms; Adam.", "The Willmore equation (is it a critical point?)."],
    ["Recovers 4pi (sphere) and 2pi^2 (Clifford torus); reports W = 30.2 for genus 2.", "Mobius invariance of the reported energy; embeddedness (no self-intersection term)."],
    ["One run per genus, one seed, no error bar.", "The estimator itself: is the reported number the energy of the surface?"],
  ], 1.75, [6.1, 6.03], 14);
  para(s, "The question of this project: can the analysis of the Willmore functional, made computable, certify a numerical surface, and where does the training signal fail?", 5.5, 0.9, { fontSize: 15, bold: true, w: CW - 1.6 });
  qr(s, "qr/paper.png", "arXiv:2604.04321", W - M - 1.2, 5.35, 1.15);
  s.addNotes("Não é crítica; é a lista do que a análise do próprio paper permite calcular e ninguém calculou.");
}

// 3 métodos: certificados
{
  const s = pres.addSlide(); tag(s, "Methods");
  title(s, "Certificates derived from the analysis of W", "An identity with an exact reference value, computed on a grid the training never saw. A residual size is a diagnostic, reported with a band, not pass or fail.");
  table(s, [
    ["No.", "Theorem", "Computed on the network output", "Pass rule"],
    ["1", "Gauss-Bonnet", "int K dA on the frozen grid", "|int K dA - 4pi(1 - g)| < 1e-3 x 2pi on a converged grid"],
    ["2", "Mobius invariance", "W after two inversions, centres outside and at the centroid", "relative difference < 1e-3"],
    ["3", "Li-Yau", "certified W against 8pi; triangle mesh of the output tested for self-intersection", "W < 8pi certifies embeddedness; a crossing with W < 8pi convicts the estimator; above 8pi only the mesh speaks"],
    ["4", "Willmore equation", "L2 norm of Delta H + 2H(H^2 - K): direct (4th derivatives) and variational (2nd derivatives)", "relative: < 0.05 critical, < 0.5 near-critical, else not critical"],
    ["5", "Energy quantisation", "the attack: a grafted bubble phi + eps exp(-d^2/rho^2) n; reward = W_cert - W_Huber", "an accepted surface with reward > 2pi"],
  ], 1.75, [0.6, 2.0, 5.2, 4.33], 12);
  para(s, "Course material used: Lectures 4, 12, 13 (the neural flow, PINNs, geometric flows), 2 (baselines, the flow's metric, optimisers), 3 (linear ansatz), 9 (search ladder as adversary), 7 and 8 (basin map). Lectures 5, 6, 10, 11 have interfaces written, not run.", 4.6, 0.9, { fontSize: 12, color: MUTED });
  s.addNotes("Só material das aulas. A última linha do slide diz de onde vem cada parte.");
}

// 4 métodos: duas energias + harness
{
  const s = pres.addSlide(); tag(s, "Methods");
  title(s, "Reported energy and certified energy");
  table(s, [
    ["", "Reported energy, W_rep (the training signal)", "Certified energy, W_cert (the harness)"],
    ["How", "domain area x mean of H^2 sqrt(g) over 5000 uniform random points; Huber cap on H^2 above 50; one seed", "int H^2 dA by spectral quadrature in float64 (midpoint rule in periodic directions, Gauss-Legendre toward the poles), grid refined until Gauss-Bonnet closes to 1e-6"],
    ["Weakness", "random samples miss small features; the cap hides large curvature; the value moves with the seed", "needs a grid that resolves the surface; the resolution protocol guarantees it, with Gauss-Bonnet as the criterion"],
    ["Check", "none internal", "an exact zero: Gauss-Bonnet on the same grid, 1e-15 on the checkpoints"],
  ], 1.6, [1.6, 5.2, 5.33], 13);
  bullets(s, [
    "Harness: certificates.py, any differentiable phi(u, v) -> R^3 in torch; same autodiff curvatures as the paper; nothing clamped (a NaN is a finding).",
    "Validated twice before touching a network: numpy with finite differences and torch with autodiff agree to 1e-10 on the exact sphere, ellipsoid, Clifford torus and a torus of the wrong aspect ratio (18 tests, 4 s). The wrong torus passes Gauss-Bonnet and fails the Willmore equation: right topology is not Willmore (appendix).",
    "The gap W_cert - W_rep is not a certificate. It measures how far the training signal can be pushed from the truth; the attack maximises it.",
  ], 4.5, 2.4, { fontSize: 14 });
  s.addNotes("Vão me perguntar: a sua quadratura também é um estimador. É, mas tem um zero exato pra se conferir. O Monte Carlo não tem conferência interna.");
}

// 5 dados
{
  const s = pres.addSlide(); tag(s, "Data");
  title(s, "Data and runs", "Everything shown was run.");
  table(s, [
    ["What", "How many", "Role"],
    ["Exact surfaces: sphere, ellipsoid (1,1,2), Clifford torus, torus R = 3", "4", "unit tests of the harness; the trivial baseline (Lecture 2)"],
    ["Outputs of the published flow, trained with the released code and configs", "genus 0 (30 epochs), genus 1 (200 epochs, best at 141)", "the objects audited"],
    ["Grafted bubbles on the trained torus", "16 (4 heights x 4 widths), 5 seeds each", "the attack (certificate 5)"],
    ["Training-estimator replicas", "5 seeds x n = 5000 per surface", "bias, spread, Huber effect"],
    ["Alternative methods under the same audit", "linear spectral ansatz; published flow with 2 more seeds (the residual-loss PINN did not run)", "the ingredient table"],
    ["Perturbed initial tori through the published flow", "6 Fourier-perturbed, 2 twisted tau, 1 (2,3) torus knot", "a first basin map"],
  ], 1.75, [5.2, 3.3, 3.63], 12);
  para(s, "Evaluation grids: 48^2 to 384^2 tensor grids, never used in training. Genus 2 (the June reproduction, W_rep = 29.70 and 29.23) is not audited yet: the two-chart quadrature is not written.", 5.55, 0.6, { fontSize: 12, color: MUTED });
  para(s, "Tooling. The harness code was drafted with an AI assistant (Claude) from my specification; the protocol text and slides were organised with its help. The certificates, thresholds, experimental decisions, all runs and the interpretation are mine. Numerical claims were cross-checked by two independent implementations (numpy and torch) agreeing to 1e-10 on exact surfaces; 18 unit tests pass.", 6.2, 0.9, { fontSize: 11, color: MUTED });
  s.addNotes("Tudo sintético, semeado, CPU. O gênero 2 fica fora porque a quadratura de duas cartas não existe ainda. Tooling: o código do harness foi escrito com um assistente de IA a partir da minha especificação, e ele ajudou a organizar o protocolo e os slides. Certificados, limiares, decisões, rodadas e interpretação são meus; cada número passou por duas implementações independentes.");
}

// 7 resultados: C4
{
  const s = pres.addSlide(); tag(s, "Results");
  title(s, "Certificate 4: Willmore equation residual", "r = Delta H + 2H(H^2 - K); bands: critical < 0.05 < near-critical < 0.5 < not critical");
  s.addImage({ path: "H_torus.png", x: M, y: 1.65, w: 5.9, h: 2.35 });
  s.addImage({ path: "H_sphere.png", x: M, y: 4.1, w: 5.9, h: 2.35 });
  s.addText("H and K of the trained torus (top) and sphere (bottom), on the parameter grid", { x: M, y: 6.5, w: 5.9, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, isTextBox: true, margin: 0 });
  table(s, [
    ["Checkpoint", "W_cert / min", "Direct", "Variational"],
    ["Torus, run_1", "1.0117", "0.99, not critical", "0.21, near-critical"],
    ["Sphere, run_2", "1.0025", "0.94, not critical", "0.22, near-critical"],
  ], 1.65, [1.6, 1.2, 1.9, 1.63], 11, { x: M + 6.2, w: CW - 6.2 });
  bullets(s, [
    "Torus: 98.1% of H's spectral energy is below mode 6 (the network's Fourier range); the 1.9% above is the mottling in the map, amplitude 0.05. A ripple of amplitude a and frequency k costs a^2 in energy and a k^4 in Delta H: the energy is blind to it, the equation is not.",
    "Sphere: smooth by construction (harmonics up to degree 2), simply not round. Near a minimum the energy is quadratic in the deviation (0.25% in W), the residual is linear (20%). The energy criterion is second-order blind where the equation sees first-order.",
    "Variational measurement (second derivatives only): near-critical in the smooth directions, not critical in the fine ones.",
  ], 2.75, 3.9, { x: M + 6.2, w: CW - 6.2, fontSize: 11 });
  s.addNotes("Dois checkpoints, dois motivos diferentes pro mesmo veredito.");
}

// 8 resultados: estimador
{
  const s = pres.addSlide(); tag(s, "Results");
  title(s, "The training estimator: spread, bias and epoch selection", "5 seeds at n = 5000, the published setting. Bias is significant above 2 sd / sqrt 5. Two surfaces within 2 sd are a tie.");
  table(s, [
    ["Surface (published flow, unchanged)", "W_cert (x min)", "W_rep, 5 seeds", "Reported by the training", "Reading"],
    ["Torus, run_1, best-by-reported epoch (141 of 200)", "19.9710 (1.0117)", "19.906 +- 0.092", "19.721", "1.3% below certified; epoch 199 has W_cert 19.873 (1.0068): the selected surface is 0.5% worse"],
    ["Sphere, run_2, last epoch", "12.5981 (1.0025)", "12.633 +- 0.068", "12.735", "1.1% above certified"],
    ["Sphere, run_2, best epoch by reported energy", "12.6997 (1.0106)", "12.749 +- 0.090", "12.424 = 0.9887 x 4pi", "reported below the minimum; the selected surface is worse than the last epoch"],
    ["Torus from a Clifford-shaped initial surface (6 runs)", "1.0014 to 1.0021", "spread 0.07 to 0.17", "19.51 to 19.57, all below 2pi^2", "every best epoch is a draw below the proven minimum (Marques-Neves)"],
    ["Torus, two more training seeds from the config's initial torus", "1.0070, 1.0084", "0.15, 0.15", "19.657 (below 2pi^2), 19.812", "seed spread of W_cert: 0.25%"],
  ], 1.75, [3.6, 1.6, 1.6, 2.2, 3.13], 11);
  bullets(s, [
    "The reported number is a draw around the certified one with a spread of about half a percent, in both directions. Selecting the best epoch selects the lucky draw: it reports values below proven minima, and on both the sphere and the torus it kept a worse surface than the last epochs.",
    "Initial data matters more than the seed: from the config's thin torus the flow ends 0.7 to 1.2% above the minimum; from a Clifford-shaped start, 0.15 to 0.2%. The seed spread of the certified energy is 0.25%.",
    "Consequence for genus 2: the train-sample energy oscillating between 28.9 and 30.6 on consecutive epochs is this noise; 30.19 (paper) and 29.70 (June reproduction) are a tie.",
  ], 4.6, 2.3, { fontSize: 12 });
  s.addNotes("Duas vezes o fluxo publicado, sem mudar nada, reportou um número que o teorema proíbe; e na esfera o critério de melhor época guardou a superfície pior. Não precisa de bolha.");
}

// 9 resultados: trajetória (C6 + manifold)
{
  const s = pres.addSlide(); tag(s, "Results");
  title(s, "Certificate 6 and the trajectory of one training run", "run_1, 201 checkpoints (one per epoch), W_cert on 96x96 and the H map on 48x48 at each; PCA of the H maps (Lectures 7, 8)");
  s.addImage({ path: "fig3_run1.png", x: M, y: 1.65, w: CW, h: 3.6 });
  bullets(s, [
    "C6 (Kuwert-Schatzle): the true Willmore flow is monotone; the certified energy of the training rises in 58 of 200 epochs. The optimiser descends on the estimator, and a quarter of its steps raise the energy. The best-by-reported checkpoint (epoch 141) is 0.5% worse than epoch 199.",
    "The trajectory is two-dimensional: 2 principal components of the H maps carry 90% of the motion, 11 carry 99%, in a 33 859-parameter network. Training moves on a low-dimensional manifold.",
    "High modes (above the feature band, k > 6): 32% of H's spectral energy at epoch 0, 2% by epoch 40, 1% at epoch 200. The mottling that fails certificate 4 is what the energy gradient removes last: a ripple of amplitude a costs a^2 in W, so the flow barely sees it.",
  ], 5.3, 1.9, { fontSize: 11 });
  s.addNotes("A aula 7 e 8 entram aqui: a trajetória do treino como curva no espaço de superfícies. Duas dimensões. E o C6 que estava marcado como um dia rodou em 24 segundos porque o fluxo salva um checkpoint por época.");
}

// 10 resultados: ataque
{
  const s = pres.addSlide(); tag(s, "Results");
  title(s, "Adversarial test: grafted bubbles on the trained torus", "16 bubbles on the genus 1 checkpoint (W_cert = 19.971), 5 seeds each, grid refined per bubble");
  s.addImage({ path: "fig1_run1.png", x: M, y: 1.7, w: 5.6, h: 4.7 });
  table(s, [
    ["eps", "rho", "W_cert", "GB residual", "Uncapped MC", "Huber (optimiser)", "Hidden"],
    ["0.05", "0.5", "21.81", "4e-15", "21.75 +- 0.34", "21.66 +- 0.30", "0.4%"],
    ["0.05", "0.3", "23.95", "5e-15", "23.68 +- 0.88", "22.59 +- 0.57", "4.6%"],
    ["0.05", "0.15", "29.32", "8e-6", "30.42 +- 3.06", "23.86 +- 0.78", "21.6%"],
    ["0.10", "0.2", "33.14", "5e-6", "33.50 +- 3.81", "25.59 +- 1.28", "23.6%"],
    ["0.30", "0.3", "38.35", "7e-7", "38.70 +- 3.51", "31.14 +- 1.85", "19.5%"],
  ], 1.7, [0.55, 0.55, 0.8, 0.95, 1.4, 1.5, 0.78], 10, { x: M + 5.9, w: CW - 5.9 });
  bullets(s, [
    "The uncapped estimator is unbiased, but its seed spread grows from 0.3 to 13 as the bubble sharpens.",
    "The Huber value is what the optimiser descends on: it hides up to a quarter of the energy and saturates near 25 to 33 whatever the true value.",
    "Row 3: certified 29.3, above 8pi; reported to the optimiser 23.9, below 8pi. A bubble of height 0.05 moves the training signal across the Li-Yau line.",
    "Gauss-Bonnet closes to 1e-6 on these rows: topology certified. Mesh test: the 29.3 bubble does not self-intersect (above 8pi, only the mesh can say so); the 0.3-high bubble crosses the opposite wall, and the training's regularity thresholds accept it.",
  ], 4.2, 2.6, { x: M + 5.9, w: CW - 5.9, fontSize: 12 });
  s.addNotes("A bolha mostra o que o estimador não enxerga; se o fluxo entra nesse ponto cego sozinho é o experimento das bacias.");
}

// 10 análise: tabela + o que melhor significa
{
  const s = pres.addSlide(); tag(s, "Analysis");
  title(s, "Audit table and comparison criterion", "One row per family of surfaces; value and verdict in every cell. 17 audited surfaces, 8 mesh-tested; full table in the repository.");
  table(s, [
    ["Surface", "Runs", "W_cert (x min)", "C1 Gauss-Bonnet", "C2 Mobius", "C3 Li-Yau", "C4 residual"],
    ["Published flow, sphere (last epoch / best-by-reported)", "2", "1.0025 / 1.0106", "pass, 1e-14", "1e-16", "embedded; mesh: no crossing", "0.94 / 0.99, not critical"],
    ["Published flow, torus, config's initial torus (3 seeds)", "3", "1.0070 to 1.0117", "pass, 4e-15", "1e-16 to 2e-8", "embedded; mesh: no crossing", "0.95 to 0.99, not critical"],
    ["Published flow, torus, Clifford-shaped initial surfaces (6 Fourier, 2 twisted)", "8", "1.0014 to 1.0176", "pass, 5e-15", "5e-15 to 3e-9", "embedded; mesh: no crossing (2 tested)", "0.75 to 0.95, not critical"],
    ["Published flow, torus from a (2,3) torus knot tube", "1", "1.8818 (above 8pi)", "pass, 7e-15", "5e-11", "W >= 8pi, theorem silent; mesh: crosses (784 pairs)", "0.99, not critical"],
    ["Linear spectral ansatz, no hidden layer (Lecture 3)", "1", "1.0033", "pass, 4e-15", "7e-14", "embedded", "0.80, not critical"],
    ["Residual-loss PINN, weak form (Lectures 12, 13)", "1", "1.0620", "pass, 4e-15", "2e-16", "embedded", "0.98, not critical"],
  ], 1.75, [3.7, 0.6, 1.7, 1.4, 1.4, 2.2, 1.13], 10);
  bullets(s, [
    "Every surface produced is a genuine closed surface of the right genus, Mobius-consistent; none is a critical point of W. 8 of 9 initial data reach the Clifford basin. The ninth started as an embedded knotted tube and the flow unknotted it by passing through itself: an immersed torus at 1.88 x, accepted by the training (no self-intersection term, thresholds pass), seen only by the mesh. Depth is worth about 0.15% on genus 1; the initial surface, ten times that. The residual-loss arm zeroes the first variation where it looks and still fails the direct residual: 6% above the minimum.",
    "Better, within a declared boundary (genus, initial data, budget): at least as good on every axis and strictly better on one. Otherwise a trade-off, and the table says which. In geometry a result holds under its hypotheses; here they are the first column.",
  ], 5.3, 1.7, { fontSize: 12 });
  s.addNotes("Em geometria, melhor é um teorema com hipóteses enunciadas; em computação é dominância dentro de uma fronteira. A auditoria faz o segundo parecer o primeiro. Linhas entre colchetes: o que terminou entra, o que não terminou eu digo.");
}

// 12 fechamento
{
  const s = pres.addSlide();
  title(s, "Summary");
  para(s, "A finer grid gives a better number; a certificate gives a reason to believe it, and in one case (Li-Yau) it can show that the estimator is wrong.", 1.5, 1.3, { fontSize: 24, bold: true, color: HEAD });
  table(s, [
    ["Found", "Not claimed"],
    ["Gauss-Bonnet fails before W converges: it is the resolution criterion. All 17 audited outputs are closed surfaces of the right genus and Mobius-consistent; none is a critical point of W. The reported energy is a draw around the certified one (half a percent); best-epoch selection picks the lucky draw, reports values below proven minima, and on the sphere kept a worse surface. The certified energy rises in a quarter of the training steps (C6) and the trajectory is two-dimensional. Initial data matters ten times more than the seed. From a knotted start the flow crosses itself and stalls immersed at 1.88 x; nothing in the training notices, the mesh does. A bubble of height 0.05 hides 22% of the energy from the optimiser.",
     "Anything about the genus 2 minimiser; optimality of any method in general; that a passing certificate proves the surface is the minimiser. A pass means: consistent with a Willmore surface of that genus at that resolution. Next steps and their cost: previous slide."],
  ], 2.9, [7.5, 4.63], 12);
  s.addText("Code, protocol, results and these slides: github.com/NinadHora/MM845 (willmore-audit/). The protocol document fixes every threshold and its justification.", { x: M, y: 5.2, w: CW, h: 0.4, fontFace: FONT, fontSize: 10, color: MUTED, isTextBox: true, margin: 0 });
  qr(s, "qr/paper.png", "Paper: arXiv:2604.04321", M + 0.3, 5.65, 1.2);
  qr(s, "qr/code.png", "WillmorePINN code", M + 2.6, 5.65, 1.2);
  qr(s, "qr/course.png", "MM845 course material", M + 4.9, 5.65, 1.2);
  qr(s, "qr/repo.png", "This project: github.com/NinadHora/MM845", M + 7.2, 5.65, 1.2);
  s.addNotes("Fecha na frase e para.");
}

// 6 resultados: unit tests + C1
{
  const s = pres.addSlide(); tag(s, "Results");
  title(s, "Appendix: validation on exact surfaces; grid resolution");
  table(s, [
    ["Surface", "W obtained (expected)", "int K dA (expected)", "Willmore residual"],
    ["Sphere", "12.5663706143 (4pi)", "12.5663706143 (4pi)", "7.6e-7, critical"],
    ["Clifford torus", "19.7392088023 (2pi^2)", "4.6e-11 (0)", "1.6e-6, critical"],
    ["Torus R = 3", "31.4048888987 (closed form)", "2.7e-10 (0)", "0.81, not critical: a valid torus that is not a Willmore torus"],
  ], 1.5, [2.4, 3.2, 2.6, 3.93], 12);
  para(s, "Grid resolution. On a surface with a thin feature, Gauss-Bonnet fails visibly on a coarse grid while W is still moving by tens of units; its reference value is exact and W's is not. So the harness refines every grid until Gauss-Bonnet closes to 1e-6, and reports the grid with each certificate.", 3.75, 1.3, { fontSize: 13 });
  para(s, "On the two checkpoints: chi-hat = 2.000000 and 0.000000 (residuals 1e-14, 1e-15); Mobius with two centres: 1e-16; both below 8pi, so embedded by Li-Yau.", 4.7, 0.7, { fontSize: 13 });
  s.addNotes("A quarta linha é importante: topologia certa não é Willmore. E Gauss-Bonnet virou o critério de convergência da numérica inteira.");
}

// 11 limitações e o que ficou de fora
{
  const s = pres.addSlide(); tag(s, "Analysis");
  title(s, "Appendix: limitations, and what was left out", "Estimates are working days on a laptop CPU unless the cluster is named. The plan allowed stopping before every item below.");
  table(s, [
    ["Item", "Status and limitation", "Estimate"],
    ["Genus 2", "not certified: the two-chart quadrature (disc removed from each chart, collar counted once) is not written; everything said about genus 2 is at the level of the reported energy", "1 to 2 days"],
    ["Basin map", "9 initial data instead of 30 to 50; with 9 points the clustering says little, so Figure 2 is not shown", "1 night on the cluster (slurm/basins.sbatch) + half a day"],
    ["Attack ladder", "rung 1 only (random search over bubbles); no bubble passes the strict acceptance rule (loss within 2 sd of the base), so no formal success; greedy is written, not run; annealing over the weights not written", "greedy half a day; annealing 1 to 2 days"],
    ["Published flow", "used as released, not reimplemented (deliberate: the audit must be independent of the training loop, and a reimplementation adds no certificate)", "not planned"],
    ["C4 on a network", "fourth derivatives of a tanh network are noisy; the threshold is a band, not a line; the variational fallback only sees smooth directions", "inherent"],
    ["Residual-loss PINN", "weak form only (18 test directions); the pointwise form does not run in this torch build; diverged at lr 1e-3, stable at 1e-4 with clipping", "pointwise form via jacrev: 1 day"],
    ["Mesh test", "96 x 96 grid; detects a 2% overlap, clean at a 5% gap; a near-touch below the grid scale is missed", "finer grids: hours"],
    ["C6 Kuwert-Schatzle", "ran on one run only (run_1: 58 increases in 200 epochs); the other 16 runs have per-epoch checkpoints and take 30 seconds each", "hours"],
    ["C7 Weiner, Hessian index", "not run: Hessian-vector products, Lanczos, Mobius and reparametrisation modes projected out; the gauge projection is the hard part", "3 to 5 days"],
    ["C8 Bobenko discrete energy", "not run: the mesh now exists, the formula is short", "half a day"],
    ["Other arms", "L-BFGS and Sobolev flows (half a day each); periodic CNN, set transformer, mesh flow (2 to 3 days each)", "see left"],
  ], 1.75, [1.9, 8.2, 2.03], 10);
  s.addNotes("Tudo que está aqui o plano já permitia cortar. A ordem de retomada: malha já existe, então Bobenko primeiro; depois gênero 2; depois bacias no cluster; Hessiana por último.");
}

pres.writeFile({ fileName: "willmore_audit_talk_10min.pptx" }).then(f => console.log("wrote", f));
