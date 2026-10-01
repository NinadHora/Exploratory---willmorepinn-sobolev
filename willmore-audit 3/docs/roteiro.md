# Roteiro da Fala

1 de outubro de 2026

Dez minutos de fala em português sobre onze slides em inglês, mais dois de apêndice que só aparecem se perguntarem, mais dois minutos de perguntas. Mil e trezentas palavras: a 140 por minuto dá nove minutos, com folga pra respirar e apontar. Os números estão nos slides: a fala interpreta, não lê. O tempo entre parênteses em cada slide é o alvo; se passou, pula pro próximo sem terminar a frase. A cola de pronúncia no fim é pra quando a banca apontar pra um símbolo.

## Problema (slides 1 e 2, um minuto e meio)

**1. Capa e o problema de Willmore (40 s).** "A energia de Uílmor mede quanto uma superfície fechada está dobrada: a integral do quadrado da curvatura média. A esfera redonda é a menos dobrada, com energia quatro pi. Entre os toros, Uílmor conjecturou que o mínimo é o toro de Clíford, dois pi ao quadrado; Marques e Neves provaram em dois mil e catorze. Com dois buracos o problema está aberto. Duas propriedades vão importar: a energia é invariante por Mébius, e os pontos críticos satisfazem uma equação de quarta ordem, a equação de Uílmor."

**2. O método auditado e a pergunta (50 s).** "O paper dos professores minimiza essa energia com uma rede neural: a rede é a parametrização da superfície, e o treino desce a energia estimada por Monte Carlo em cinco mil pontos, com um cap na curvatura e termos de regularidade. Recupera a esfera e o Clíford e produz uma superfície de gênero dois. Quando foi apresentado aqui, a pergunta que ficou foi: como a gente sabe que esse número é verdade? Na saída nada é conferido: nem a topologia, nem a equação, nem a invariância por Mébius, nem se a superfície se cruza, e o estimador tem uma semente só. O projeto pega isso ao pé da letra: a análise do funcional, tornada computável, consegue certificar uma superfície numérica? E onde o sinal de treino falha?"

## Métodos (slides 3 e 4, um minuto e meio)

**3. Certificados (50 s).** "Cada teorema sobre a energia diz algo que toda superfície verdadeira satisfaz, com valor exato. Então cada teorema vira uma conta na saída da rede, numa grade que o treino nunca viu. Gáus-Boné confere a topologia. Mébius é um termômetro: se eu inverto e a energia muda, o erro é do estimador, nunca da geometria. Li-Iáu: abaixo de oito pi a superfície não pode se cruzar; e agora também uma malha triangular testa o cruzamento diretamente. A equação de Uílmor diz se é ponto crítico ou se parou num declive. E a quantização de energia vira o ataque: uma bolha pequena custa quase quatro pi e ocupa quase nenhuma área de parâmetro, então o Monte Carlo não vê. Certificado é só o que tem referência exata; resíduo é diagnóstico, com faixa. Tudo material das aulas."

**4. Reportada e certificada (35 s).** "Daqui pra frente, dois números que eu nunca misturo. A energia reportada é o que o treino calcula: Monte Carlo com cap. A energia certificada é quadratura de alta precisão numa grade refinada até Gáus-Boné fechar, e ele fecha a precisão de máquina. Validei o harness de dois jeitos independentes nas superfícies exatas, e os dois concordam até a décima casa. A diferença entre as duas energias não é certificado: é quanto o sinal de treino pode se afastar da verdade, e é o que o ataque maximiza."

## Dados (slide 5, meio minuto)

**5. Dados (30 s).** "Tudo sintético, semeado, numa CPU de laptop. Superfícies exatas como testes unitários; dois checkpoints do fluxo publicado que eu mesma treinei com o código deles; bolhas enxertadas; réplicas do estimador com várias sementes; e, rodados esta noite, os métodos alternativos e nove dados iniciais perturbados. Gênero dois fica fora: a quadratura de duas cartas não está escrita."

## Resultados (slides 6 a 9, uns quatro minutos)

**6. A equação de Uílmor (65 s).** "Primeiro resultado: os dois checkpoints têm energia perto do mínimo e os dois falham na equação, por motivos diferentes. No toro, em cima, a estrutura grande está certa, mas tem um mosqueado fino: ondulações de uns três por cento em frequências acima do que as features da rede alcançam. Uma ondulação dessas custa quase nada em energia, porque entra ao quadrado da amplitude, mas entra na equação com a quarta potência da frequência. A energia é cega a isso; a equação não. Na esfera, embaixo, não tem ondulação: a rede é lisa por construção, só não é redonda. Perto de um mínimo, a energia é quadrática no desvio e o resíduo é linear: um quarto de por cento em energia vira vinte por cento no resíduo. O critério de energia é cego em segunda ordem onde a equação enxerga em primeira."

**7. O estimador (55 s).** "Repeti o estimador do treino com cinco amostras. O viés não é significativo, mas o número que o treino guarda como melhor é um sorteio em torno do valor certificado, com meio por cento de dispersão. Escolher a melhor época pela energia reportada escolhe o sorteio favorável, e isso apareceu três vezes. Na esfera, a época escolhida reportou um valor abaixo de quatro pi, que nenhuma esfera pode ter, e é pior que a última época. No toro, a época escolhida é meio por cento pior que a do fim do treino. E partindo de superfícies com a forma do Clíford, as seis rodadas reportaram como melhor um valor abaixo do mínimo que Marques e Neves provaram. Outra coisa: o dado inicial importa dez vezes mais que a amostra."

**8. A trajetória de um treino (45 s).** "O fluxo salva um checkpoint por época, então dá pra olhar um treino inteiro como uma curva. À esquerda, o certificado seis: o fluxo de Uílmor de verdade decresce a energia sempre; aqui a energia certificada sobe em um quarto das épocas, porque o otimizador desce no estimador. No meio, componentes principais dos mapas de curvatura, aulas sete e oito: duas componentes carregam noventa por cento do movimento, numa rede de trinta e três mil parâmetros; o treino anda num plano. À direita, a fração da curvatura nos modos altos cai de trinta pra dois por cento cedo e depois quase não mexe: o mosqueado que falha na equação é o que o gradiente da energia remove por último."

**9. O ataque (55 s).** "Enxertei bolhas no toro treinado. Eixo horizontal, energia certificada; vertical, o que o treino reporta. O estimador sem cap fica na diagonal, mas as barras de erro explodem conforme a bolha afina. O valor com cap, em laranja, é o que o otimizador de fato desce, e ele satura: por mais energia que a bolha tenha, o otimizador vê mais ou menos o mesmo. A linha que importa: uma bolha de altura cinco centésimos leva a energia certificada acima de oito pi enquanto o otimizador vê abaixo de oito pi. O sinal de treino cruzou a linha de Li-Iáu, e a malha confirma que a superfície não se cruza: o erro é do estimador. Já a bolha mais alta atravessa a parede oposta, e os limiares do treino aceitam ela do mesmo jeito."

## Análise (slides 10 e 11, dois minutos)

**10. Tabela e critério (70 s).** "Uma linha por família; dezessete superfícies, oito pela malha, tabela completa no repositório. Três leituras. Primeira: tudo que o método produziu é superfície fechada do gênero certo, consistente com Mébius, e nenhuma é ponto crítico. Segunda: oito dos nove dados iniciais caem na bacia do Clíford. O nono partiu de um tubo nodado mergulhado, e o fluxo desatou o nó atravessando a si mesmo: parou num toro imerso, quase no dobro do mínimo, aceito pelo treino; só a malha vê. Terceira, os ingredientes: o ansatz linear sem camada escondida chega a três décimos por cento do mínimo; a rede profunda, de partida comparável, a um ou dois décimos. A profundidade vale pouco; a superfície de partida vale dez vezes mais. O PINN de resíduo fica crítico só onde o loss testou, seis por cento acima do mínimo. Sobre qual método é melhor: em geral nenhum. Dentro de uma fronteira declarada, gênero, dados iniciais, orçamento, um método é melhor se é pelo menos igual em todos os eixos e melhor em um. Em geometria um resultado vale sob hipóteses; aqui as hipóteses são a primeira coluna."

**11. Resumo (30 s).** "Uma grade mais fina dá um número melhor; um certificado dá um motivo pra acreditar nele, e em um caso ele mostra que o estimador está errado. Não afirmo nada sobre gênero dois, nem otimalidade em geral; um certificado que passa diz só que a saída é consistente com uma superfície de Uílmor daquele gênero naquela resolução. Limitações, o que ficou de fora e o custo de cada coisa estão no apêndice. Obrigada."

## Apêndice (só se perguntarem)

**A1. Validação e resolução.** "Nas superfícies exatas tudo fecha a precisão de máquina. A linha que interessa é a última: um toro com a proporção errada passa em Gáus-Boné e falha na equação. Topologia certa não quer dizer superfície de Uílmor. E numa superfície com detalhe fino, a curvatura total, que tem valor exato, falha numa grade grossa enquanto a energia ainda muda dezenas de unidades; por isso Gáus-Boné decide se a grade resolve a superfície."

**A2. Limitações e o que ficou de fora.** "Tudo aqui o plano permitia cortar; as estimativas são dias de laptop. Gênero dois não certificado: a quadratura em duas cartas, um ou dois dias. Bacias com nove dados iniciais em vez de trinta a cinquenta: uma noite no cluster. Ataque só no primeiro degrau, e nenhuma bolha passa na regra estrita de aceitação, então não há sucesso formal. Dos certificados: Kuvert-Chétsle rodou em um treino só; a Hessiana de Weiner, três a cinco dias pela projeção dos modos de gauge; Bobênko, meio dia agora que a malha existe. Duas limitações inerentes: quartas derivadas de rede tanh são ruidosas, por isso faixa; e a malha não vê quase-toque menor que a grade. E uma escolha: usei o fluxo publicado como está, porque a auditoria tem de ser independente do treino."

## Cola de pronúncia

Regra: fala o significado, não o símbolo. A banca lê o símbolo no slide; você dá o nome dele.

| No slide | Como falar |
| --- | --- |
| W, W\_cert, W\_rep | "a energia de Uílmor", "a energia certificada", "a energia reportada" |
| H, K | "a curvatura média", "a curvatura gaussiana" |
| int H² dA | "a integral de H ao quadrado na área" |
| int K dA, 2πχ | "a curvatura total"; "dois pi vezes a característica de Euler" (χ é "qui") |
| 4π(1 − g) | "quatro pi vezes um menos o gênero" |
| ΔH + 2H(H² − K) | "o laplaciano de H mais dois H vezes H ao quadrado menos K"; ou só "a equação de Uílmor" |
| φ(u, v) → R³ | "fi de u e v no R três" |
| Ψ ∘ φ | "a superfície invertida" |
| ε, ρ | "épsilon" (altura da bolha), "rô" (largura) |
| n, r, ξ | "a normal", "o resíduo", "csi" (a função-teste) |
| ±, sd | "mais ou menos", "o desvio entre sementes" |
| k, k⁴ | "o modo k", "k à quarta" |
| √g, dA | "o elemento de área" |
| 4π = 12.566 | "quatro pi, doze vírgula cinco seis" |
| 2π² = 19.739 | "dois pi ao quadrado, dezenove vírgula sete quatro" |
| 8π = 25.13 | "oito pi, vinte e cinco vírgula um" |
| 1.0117, 1.0025 | "um vírgula dois por cento acima do mínimo", "zero vírgula vinte e cinco por cento acima" |
| 19.971 | "dezenove vírgula noventa e sete" |
| 29.70, 29.23, 30.19 | "vinte e nove e setenta", "vinte e nove e vinte e três", "trinta e dezenove" |
| 1e-15, 1e-6, 1e-3 | "dez elevado a menos quinze" (ou "precisão de máquina"), "a menos seis", "a menos três" |
| 0.99, 0.21 | "zero vírgula noventa e nove", "zero vírgula vinte e um" |
| ±0.092 | "mais ou menos meio por cento" |
| 192², 384² | "cento e noventa e dois ao quadrado", "trezentos e oitenta e quatro ao quadrado" |
| n = 5000 | "cinco mil pontos" |
| h2\_clip = 50, | H |

**Nomes**

| Nome | Pronúncia |
| --- | --- |
| Willmore | Uílmor |
| Gauss-Bonnet | Gáus-Boné (o t é mudo) |
| Möbius | Mébius |
| Blaschke | Bláchke |
| Li-Yau | Li-Iáu |
| Marques-Neves | como em português |
| Kuwert-Schätzle | Kúvert-Chétsle |
| Rivière | Riviér |
| Simon | Sáimon |
| Weiner | Uáiner |
| Bobenko | Bobênko |
| Lawson | Lóson |
| Clifford | Clíford |
| Huber | Húber |
| Sobolev | Sóbolev |
| AdamW, L-BFGS | "Adam dáblio", "éle bê éfe gê ésse" |
| DBSCAN, kernel PCA | "dê bê scan", "kernel pê cê á" |
| autodiff, jacfwd | "diferenciação automática", "forward-mode" |

**Três atalhos**

- Em vez de "resíduo relativo zero vírgula noventa e nove, faixa não crítico": "o resíduo é da ordem da própria escala: não é ponto crítico".
- Em vez de "delta relativo um ponto sete e menos dezesseis": "igual até a última casa".
- Em vez de "W\_rep com Huber": "o que o otimizador vê".

Se travar num símbolo, aponta pro slide e diz o que ele mede. Ninguém vai cobrar a leitura da fórmula; vão cobrar se você sabe o que ela significa.

## Se perguntarem

**A sua quadratura também é um estimador.** "É, mas tem um zero exato pra se conferir: Gáus-Boné na mesma grade dá precisão de máquina, e eu só aceito uma grade em que ele fecha. O Monte Carlo não tem conferência interna; a única referência dele é a quadratura."

**O cap de Húber não muda o minimizador. Por que conta como defeito?** "Não muda onde o mínimo está; muda o que o otimizador vê no caminho. Uma bolha com H acima de sete é cobrada linearmente em vez de quadraticamente, e a superfície de vinte e nove vírgula três reporta vinte e três vírgula nove. É uma escolha de treino defensável e uma escolha de relatório indefensável; a auditoria separa as duas."

**A bolha é artificial. O fluxo produz uma sozinho?** "Essa é a segunda metade do problema, e é o experimento dos dados iniciais perturbados. A bolha mostra o que o estimador não enxerga. E o fluxo entra num ponto cego sozinho, sem enxerto: partindo do tubo nodado ele atravessa a si mesmo e para num toro imerso que o treino aceita. Não é a bolha, é outro ponto cego, a auto-interseção, e só a malha vê."

**Quartas derivadas de uma rede tanh por diferenciação automática fazem sentido?** "Nas superfícies exatas o resíduo dá dez a menos seis, então o operador está certo. Na rede ele mistura a geometria com a aspereza da própria rede; por isso o limiar é uma faixa, não uma linha, e por isso existe a medição variacional, só com segundas derivadas, que dá zero vírgula dois nos dois checkpoints."

**Por que o gênero dois não está nos certificados?** "Duas cartas coladas precisam de uma quadratura que tire o disco de cada carta e conte a costura uma vez. Não é obstáculo conceitual, é código que ainda não escrevi. Tudo que eu digo sobre gênero dois hoje se apóia na energia reportada e no warm restart de junho, que é uma afirmação no nível do estimador."

**Por que cinco sementes e cinco mil pontos?** "Porque é o setting publicado; eu quero medir o estimador como ele é usado, não um melhor. Cinco sementes dão um desvio grosseiro, e a regra de empate usa dois desvios pra absorver isso."

**Li-Iáu só limita a multiplicidade. Isso é certificado?** "Pro gênero um é completo: energia certificada abaixo de oito pi força mergulho. Pro gênero dois, a vinte e nove e sete, não diz nada, e eu digo isso. O teste de malha que falaria pro gênero dois é o próximo passo."

**Qual método é melhor?** "Em geral, nenhum: no free lunch. Dentro de uma fronteira declarada, gênero, dados iniciais, orçamento, um método é melhor se domina o outro nos seis eixos. A tabela de ingredientes diz de qual ingrediente cada garantia depende, e a fronteira está na primeira coluna."

**Em que isso difere de só usar uma grade mais fina?** "Uma grade mais fina dá um número melhor; não dá um motivo pra acreditar nele. Os certificados dão motivos com valor de referência exato, e a linha de contradição do Li-Iáu pode condenar o estimador, coisa que refinar não faz."

**O que do curso você usou?** "Os certificados são análise tornada computável por diferenciação automática, aulas quatro, doze e treze. Os ataques são a escada de busca da aula nove. A análise de componentes principais da trajetória e o mapa de bacias são sete e oito. Os ansätze alternativos são três, cinco, seis, dez e onze. A demo de Plateau é a linha de base em malha."

**Por que o PINN de resíduo em forma fraca e não a equação ponto a ponto?** "A forma forte precisa de quartas derivadas da rede e do gradiente nos parâmetros por cima delas; na versão do torch que eu tenho isso não roda. A forma fraca força a primeira variação de W a zero em dezoito direções suaves, com as mesmas segundas derivadas do fluxo publicado. É a medição variacional do certificado quatro usada como loss."

**Os certificados do PINN são os mesmos dos outros?** "Exatamente os mesmos: mesmo harness, mesma grade, mesmos limiares. Só mudou a superfície que entrou. É isso que faz as linhas da tabela comparáveis. Uma ressalva: o resíduo variacional dessa superfície dá pequeno, mas foi o que o treino minimizou, então não conta como evidência; o número da tabela é o resíduo direto, de quartas derivadas, que o treino nunca viu."
