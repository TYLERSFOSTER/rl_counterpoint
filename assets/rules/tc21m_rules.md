# Counterpoint rules & guidelines
The following rules come form the book [*Tonal Counterpoint for the 21st-Century Musician*]() by Teresa Davidian.

***[FIX THIS...]*** The basic pattern here is that the *$n^{\text{th}}$-species* means "counterpoint for $n+1$ voices. The shift in indices here is actually convenient from a combinatorial/categorical perspective, because it means the system of possible relationsips in $n^{\text{th}}$-species is represented by an $n$-simplex:
## $0^{\text{th}}$-species: the lone singer, ie.., a *melody*
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/1st_species_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images/1st_species_light.png">
    <img src="images/1st_species_light.png" alt="description" width="310" style="opacity: 0.65;">
  </picture>
</p>

### Single jumps
- Intervals should be, *most often*, **consonant**. Specifically, they should come from the set $$\{\text{m3},\;\text{M3},\;\text{P4},\;\text{P5},\;\text{M6},\;\text{P8}\}.$$
  - **COMPUTATIONAL NOTE:** This refers to horizontal intervals, i.e., to the individual component actions. The $k^{\text{th}}$ component of the total $\Delta s$ must come from $\{3, 4, 5, 7,8,9\}\;(\!\!\!\mod 12)$... or is that $7,9,10$?... I might have these numbers wrong. Regardless, the point is that the reward here should check that the consant-to-dissonant distribution in the rank $1$ window $W^{1}_{t}$ closely matches a preset one like maye $(\text{consonant};\:\text{disonant})=(80:20)$ over $W^{1}_{t}$.
- Dissonant intervals should be used sparingly. These are the intervals $$\{\text{m2},\;\text{M2},\;\text{A4/d5},\;\text{m6},\;\text{m7},\;\text{M7}\}.$$
  - 7ths and also large consonant jumps should be subsequently moved in the opposite direction, with a kind of switchback.
    - **COMPUTATIONAL NOTE:** Here I imagine a reward term that is the weighted product of two values, gotten from **(i)** a Boolean check that absolute value $\left|\Delta s^{1}_{t-1}\right|$, i.e., the distance $\left|s^{1}_{t}-s^{1}_{t-1}\right|$, is sufficiently large, like say $6$ or larger, and **(ii)** a Boolean check that the proposed action $\Delta s^{1}_{t}$ moves in the opposite direction, i.e., that $$\operatorname{sgn}(\Delta s^{1}_{t-1})-\operatorname{sgn}(\Delta s^{1}_{t})\;\equiv\;0\;(\!\!\!\!\!\mod 2).$$
  - Diminished and augmented intervals need to be subsequently "resolved" by moving up or down a single scale degree
    - **COMPUTATIONAL NOTE:** This term should be computed in much the same way a the one immediately above, except that the Booleans are now **(i)** a check that $\left|\Delta s^{1}_{t-1}\right|$ is a diminished or augmented interval, and **(ii)** a check that the move $s^{1}_{t-1}\mapsto s^{1}_{t}$ is a move up or down consecutive scale degrees... Concretely, **(ii)** could be a check that $s^{1}_{t-1}$ and $s^{1}_{t}$ are both inour underlying scale, with $\left|\Delta s^{1}_{t-1}\right|\le 2$ (ensures consecutive).
### Consecutive jumps
- Avoid jumps that give the "recent" melody a range of more than an octave.
  - **COMPUTATIONAL NOTE:** A weighted Boolean reward term checking that $$\max\{s^{1}_{t-4},\;\dots,\;s^{1}_{t-1}\}-\min\{s^{1}_{t-4},\;\dots,\;s^{1}_{t-1}\}\;\le\;12.$$
- Avoid consecutive 4ths and 5ths.
  - **COMPUTATIONAL NOTE:** A weighted term that does a check that if $\Delta s^{1}_{t-2}\;=\;\Delta s^{1}_{t-1}$, then $\Delta s^{1}_{t-1}\notin\{5,7\}$.
### Steps and jumps
- Follow large leaps with *small* step in opposite direction.
    - **COMPUTATIONAL NOTE:** Weighted term checking that if $\left|\Delta s^{1}_{t-1}\right|\ge 6$, then $\left|\Delta s^{1}_{t}\right|\le 3$. That it should also be in the opposite direction is already taken care of by a rule above.
- Avoid stepwise motion followed by a leap in the opposite direction.
    - **COMPUTATIONAL NOTE:** Weighted term checking the condition that if $\left|\Delta s^{1}_{t-1}\right|\le 3$ *and* if $$\operatorname{sgn}(\Delta s^{1}_{t-1})-\operatorname{sgn}(\Delta s^{1}_{t})\;\equiv\;0\;(\!\!\!\!\!\mod 2),$$ then $\left|\Delta s^{1}_{t}\right|\le 6$.
### Scale
- Give emphasis to the tonic scale degree.
    - **COMPUTATIONAL NOTE:** Define the *step sequence* associated to a given window $W^{k}_{t}$ of any rank $k$ to be the set $S_{t}:=\{t-L,\;\dots,\;t-2,\;t-1\}$ of time steps that the window is supported on, and define $\text{MIDI}:=\{0,1,2,\dots,127\}$. Then each window $W^{k}_{t}$, of any rank, has an associated $\mathbb{R}$-valued *characteristic function* $\mathbb{1}_{W^{k}_{t}}$ defined on $S_{t}\times\text{MIDI}$ according to
      $$
      \mathbb{1}_{W^{k}_{t}}(i,\lambda)=
      \begin{cases}
      1 & \text{if}\;\lambda\;\text{is an entry of}\;s^{k}_{i} \\
      0 & \text{otherwise} \\
      \end{cases}
      $$
      By integrating this characteristic function along the our step sequence, we get a *spectral density* function defined on our pitch space $\text{MIDI}$:
      $$
      \operatorname{spec-dens}(\lambda)
      \;:=\;\frac{1}{k\cdot\text{width}(W^{k}_{t})}\cdot\int_{S_{t}}\!\mathbb{1}_{W^{k}_{t}}(i,\lambda)\;di
      $$
      It can also be thought of as a direct image of $\mathbb{1}_{W^{k}_{t}}$ along the second coordinate projection $S_{t}\times\text{MIDI}\longrightarrow \text{MIDI}$. Be can get measures of how much certain notes show up by checking this spectral density against other distributions, like one that puts translate of tonic in some prefixed ratio, for instance by just taking dot products of distributions. The point of doing it as this integral is that everything can by put on a GPU.

- Shape the melodic line around structural pitches.
  - Put structural pitches at stronger beats
  - Avoid putting unstable scale degrees at stronger beats.
  - **COMPUTATIONAL NOTE:** I think the "structural" pitches are probably $\{0,5,7\}\;(\!\!\!\mod\;12)$. Whatever they are, this reward should just be a weighted term... that checks... maybe measures "structural/neutral/unstable" against "1/down/up".
- In ascending context, resolve the leading tone (7th) to tonic as quickly as possible.
    - **COMPUTATIONAL NOTE:** When a seventh occurs, measure how many time steps until resolved, and then reward with weighted
    $$
    \frac{1}{\;\#\text{steps until resolved}\;}.
    $$
### Range
- When moving in one direction, up or down, do not expose unstable intervals such as a $\text{A4}/\text{d5}$ or 7th.
    - **COMPUTATIONAL NOTE:** The unstable intervals are the pitches in $\lambda_{\text{tonic}}+\{2,6,11\}+12\,\mathbb{Z}$. The weighted reward term checks that *(i)* these dont occur on down beats, and that *(ii)* these *really* down happen on first beat of measures. The obvious is by subtracting a measure of how often these occur, say by some tensor convolution.
- Use V-I cadence at end of passage.
    - **COMPUTATIONAL NOTE:** This reward is maybe a bit complicated to compute because it is tangled up with *passage goal*, which hasn't been described yet. The basic "game" idea here is that you give the agent it's starting chord, and its ending octave, and it has to come up with a good voice leading passage moving from one to the other. It completes its overarchign goal if it gets to the ending octave, but this present reward is uspposed to add "in such a way that it ends the passage with a V-I on weak/strong of last measure. Some more discussion is necessary here.
## $1^{\text{st}}$-species: simple accompaniment, i.e., a *duo*
A new notion of "movement" emerges now, namely *relational* movement. Instead of the total movement of a line, we can talk about th relative movement of the two lines. 
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/2nd_species_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images/2nd_species_light.png">
    <img src="images/2nd_species_light.png" alt="description" width="325" style="opacity: 0.65;">
  </picture>
</p>
In what follows, it is important to realize that if $n>1$, then the entry index $1$ that appears in this discussion of $2$-part voiceleading is really the outer voice, i.e., voice $n-1$. I use the index $1$ here in an attempt to avoid confusion.

### Melodic motion
- Use variety of expansion versus contraction.
  - **COMPUTATIONAL NOTE:** Define the *midpoint* and *radius* of any vertical interval, i.e., any $2$-chord $(\lambda_0,\,\lambda_1)$, at time step $t$, to be the rational numbers
    $$
    p_{t}\;:=\;\frac{\;\lambda_0+\lambda_1\;}{2}
    \qquad\text{and}\qquad
    r_{t}\;:=\;\frac{\;\left|\lambda_1-\lambda_0\right|\;}{2}.
    $$
    These come with corresponding single-step differentials $\Delta p_{t}$ and $\Delta r_{t}$. Parallel, similar, and contrary motion are classified by the sign of the latter:
    $$
    \operatorname{sgn}\;\Delta r_{t}
    \;=\;
    \begin{cases}
    +1 & \text{if motion at step}\;t\;\text{expands} \\
    0 & \text{if motion at step}\;t\;\text{is parallel} \\
    -1 & \text{if motion at step}\;t\;\text{contracts} \\
    \end{cases}
    $$
    The reward term here computes that value of $\operatorname{sgn}\;\Delta r_{t}$ is well distributed over $\{-1,0,+1\}$.
- Use a variety of parallel, similar, and contrary motion.
  - **COMPUTATIONAL NOTE:** Parallel motion occurs when $\Delta\lambda_0=\Delta\lambda_1$
    $$
    \big|\;\text{sgn}\;\Delta \lambda_{1}-\text{sgn}\;\Delta \lambda_{0}\;\big|
    \;:=\;
    \begin{cases}
    2 & \text{if motion at step}\;t\;\text{is contrary} \\
    1 & \text{if motion at step}\;t\;\text{is oblique} \\
    0 & \text{if motion at step}\;t\;\text{is similar} \\
    \end{cases}
    $$
    I'm being lazy: there are obvious ways to combine this and previous two get this weighted measure.
- Avoid *oblique* motion, that is, motion where one voice moves but the other is stationary.
  - **COMPUTATIONAL NOTE:** See computational note immediately above.
- Avoid "immediate" (how immediate?) repetiion of chords and notes.
  - **COMPUTATIONAL NOTE:** The convolution of characteristic functions
  $$
  (\mathbb{1}_{W^{k}_{t}}\ast \mathbb{1}_{W^{k}_{t}})(i,\lambda)
  \;=\;
  \int_{S_{t}}\mathbb{1}_{W^{k}_{t}}(i,\lambda)\cdot\mathbb{1}_{W^{k}_{t}}(j-i,\lambda)\;dj
  $$
  on $S_{t}\times\text{MIDI}$ can be used to detect repeated notes and chords in the obvious ways.
### Vertical intervals
- Use only consonances for outer intervals in chord (in $1^{\text{st}}$-species, this is whole chord).
  - **COMPUTATIONAL NOTE:** This is just the check that $\lambda_{1}-\lambda_0\in\{3, 4, 5, 7, 8, 9\}\;(\pmod 12)$ (or whatever the set of consonances is).
- Common occurances in vertical intervals:
  - $\text{P8}$ over tonic at beginning or end of phrase
  - $\text{P5}$ over dominant pitches at or near beginning or end of phrase.
  - $\text{P8}$ can appear over dominant approximately midway through phrase, surrounded by $3$s and $6$s.
  - **COMPUTATIONAL NOTE:** [...]
- Between these loci in the phrase, imperfect cadences should be used to maintain sense of flow.
  - Parallel $3$rds and $6$ths are ok, be more than 4 successive is excessive (destroys sense of line indpendence)
  - $\text{P5}$ and $\text{P8}$ are ok at these places, but only if surrounded by $\text{3}$ rds and $\text{6}$ ths.
  - **COMPUTATIONAL NOTE:** [...]
- Avoid parallel perfect intevals.
  - **COMPUTATIONAL NOTE:** [...]
### Harmonic succesions
- **COMPUTATIONAL NOTE:** For all of these, the basic pattern is to take the vertical interval $\pmod{12}$, minus $\lambda_{\text{tonic}}$ and then do convolutional checks for these progreesions, since the progressions are defined as characteritic functions on
  $$
  \{t-1,\;...,\;t-L\}\times\text{MIDI},
  $$
  where $L$ is the length of the cadence. For instance, we can check for $\text{I}\to\text{V}$ and $\text{ii}\to\text{V}\to\text{I}$ cadences by convolving with the shape `[2,12]` and `[3,12]` tensors
  $$
  \begin{array}{rccc}
  11:& \square \!\!\!&\!\!\! \square \\[-3pt]
  10:& \square \!\!\!&\!\!\! \square \\[-3pt]
  9:& \square \!\!\!&\!\!\! \blacksquare \\[-3pt]
  8:& \square \!\!\!&\!\!\! \square \\[-3pt]
  7:& \square \!\!\!&\!\!\! \square \\[-3pt]
  6:& \square \!\!\!&\!\!\! \square \\[-3pt]
  5:& \square \!\!\!&\!\!\! \blacksquare \\[-3pt]
  4:& \blacksquare \!\!\!&\!\!\! \square \\[-3pt]
  3:& \square \!\!\!&\!\!\! \square \\[-3pt]
  2:& \square \!\!\!&\!\!\! \square \\[-3pt]
  1:& \square \!\!\!&\!\!\! \square \\[-3pt]
  0:& \blacksquare \!\!\!&\!\!\! \square \\[-3pt]
  \end{array}
  \quad\text{and}\quad
  \begin{array}{rccc}
  11:& \square \!\!\!&\!\!\! \blacksquare \!\!\!&\!\!\! \square \\[-3pt]
  10:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  9:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  8:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  7:& \square \!\!\!&\!\!\! \blacksquare \!\!\!&\!\!\! \square \\[-3pt]
  6:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  5:& \blacksquare \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  4:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \blacksquare \\[-3pt]
  3:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  2:& \blacksquare \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  1:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \square \\[-3pt]
  0:& \square \!\!\!&\!\!\! \square \!\!\!&\!\!\! \blacksquare \\[-3pt]
  \end{array}
  $$
  respectively. Actually implement a specific positive or negative reward here is then just a matter fo figuring out what the detecting tensor shoiuld be for a given cadence, and when/how it should be used.
- Common chord *PROgressions* should predominate:
  - *descending fifths*:
    - $\text{I}\to\text{IV}$
    - $\text{iii}\to\text{vi}$
    - $\text{V}\to\text{I}$
    - $\text{ii}\to\text{V}\to\text{I}$
  - *descending thirds*:
    - $\text{i}\to\text{VI}$
    - $\text{vii}^{\circ}\to\text{V}\to\text{III}^{+}$
  - *ascending seconds*:
    - $\text{I}\to\text{ii}\to\text{iii}$
    - $\text{iv}\to\text{V}\to\text{VI}\to\text{vii}^{\circ}$
  - **COMPUTATIONAL NOTE:** [...]
- Avoid *RETROgressions*:
  - ascending thirds:
    - $\text{VI}\to\text{i}$
    - $\text{III}^{+}\to\text{V}\to\text{vii}^{\circ}$
  - descending seconds:
    - $\text{iii}\to\text{ii}\to\text{I}$
    - $\text{vii}^{\circ}\to\text{VI}\to\text{V}\to\text{iv}$
  - ***completely avoid***:
    - $\text{V}\to\text{IV}\;\text{(iv)}$
  - **COMPUTATIONAL NOTE:** [...]
- Avoid retrogressions when approaching the cadence.
  - **COMPUTATIONAL NOTE:** [...]
- Use lots of $\text{3}$ rds for vertical intervals
  - **COMPUTATIONAL NOTE:** [...]
### Cadences
- End with a cadence
  - **COMPUTATIONAL NOTE:** [...]
- For $1^{\text{st}}$-species, use a perfect cadence
  - **COMPUTATIONAL NOTE:** [...]
## $2$-simplices: three's a crowd, ie., a *trio*
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/triadic_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images/triadic_light.png">
    <img src="images/triadic_light.png" alt="description" width="340" style="opacity: 0.65;">
  </picture>
</p>

### Vertical intervals
- Dissonant vertical intervals occur more often on offbeat.
  - **COMPUTATIONAL NOTE:** [...]
- Dissonant vertical intervals can occur on onbeat only rearely, and only if follwed by a vertical consonance.
  - **COMPUTATIONAL NOTE:** [...]
- Consonant intervals should occur $>$ half the time among all vertical intervals (not clear here if "all vertical" means all $(n+1)$-choose-$2$ edges in the $n$-simplex $\Delta^{n}$)
  - **COMPUTATIONAL NOTE:** [...]

### Melodic motion
- Pedal in $2^{nd}$-species is allowed, and can even be encouraged, to repeat for down/off beat pairs.
  - **COMPUTATIONAL NOTE:** [...]
- Non-harmonic tones:
  - Good balance between chord tones and non-chord tones.
    - **COMPUTATIONAL NOTE:** [...]
  - Move stepwise to and from disonant pitches, so that most dissonances are neighboring or passing tones.
    - **COMPUTATIONAL NOTE:** [...]
  - unaccented passing tones should occur more often than other dissoances.
    - **COMPUTATIONAL NOTE:** [...]
  - Accentented non-harmonic tones shoul dbe followed by a chord tones.
    - **COMPUTATIONAL NOTE:** [...]
  - The only *skips* and *leaps* in $2^{\text{nd}}$-species occur between chord tones.
    - **COMPUTATIONAL NOTE:** [...]
  - The above *consonant/chordal skips and leaps* occur freely, on down and offbeats.
    - **COMPUTATIONAL NOTE:** [...]

### Harmonic considerations
- Do not ommit the root or third of a chord, be it triad or seventh
  - **COMPUTATIONAL NOTE:** [...]
- Triads should be implied in one of the following ways:
  - Include all 3 chord factors: root, third, and fifth
  - Write 2 roots and one third
  - **COMPUTATIONAL NOTE:** [...]
- When a non-harmonic tone is involved, it is preferable to include root and third factor
  - **COMPUTATIONAL NOTE:** [...]
- Do not double the leading tone of the scale **[SHOULD BE PRUNING RULE]**
  - **COMPUTATIONAL NOTE:** [...]
- Seventh chords should be implied by root, third, and seventh
  - **COMPUTATIONAL NOTE:** [...]
- Six-four chords should be one of these three:
  - cadential
  - passing
  - arpegiating
  - **COMPUTATIONAL NOTE:** [...]
  

## $3$-simplices: the barber shop *quartet*
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/tetra_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images/tetra_light.png">
    <img src="images/tetra_light.png" alt="description" width="350" style="opacity: 0.65;">
  </picture>
</p>

## $n$-simplices'

[...]