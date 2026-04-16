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
    - **COMPUTATIONAL NOTE:** Here I imagine a reward term that is the weighted product of two values, gotten from **(i)** a Boolean check that absolute value $\left|\Delta s^{1}_{t-1}\right|$, i.e., the distance $\left|s^{1}_{t}-s^{1}_{t-1}\right|$, is sufficiently large, like say $6$ or larger, and **(ii)** a Boolean check that the proposed action $\Delta s^{1}_{t}$ moves in the opposite direction, i.e., that $$\operatorname{sgn}(\Delta s^{1}_{t-1})-\operatorname{sgn}(\Delta s^{1}_{t-1})\;\equiv\;0\;(\!\!\!\!\!\mod 2).$$
  - Diminished and augmented intervals need to be subsequently "resolved" by moving up or down a single scale degree
    - **COMPUTATIONAL NOTE:** This term should be computed in much the same way a the one immediately above, except that the Booleans are now **(i)** a check that $\left|\Delta s^{1}_{t-1}\right|$ is a diminished or augmented interval, and **(ii)** a check that the move $s^{1}_{t-1}\mapsto s^{1}_{t}$ is a move up or down consecutive scale degrees... Concretely, **(ii)** could be a check that $s^{1}_{t-1}$ and $s^{1}_{t}$ are both inour underlying scale, with $\left|\Delta s^{1}_{t-1}\right|\le 2$ (ensures consecutive).
### Consecutive jumps
- Avoid jumps that give the "recent" melody a range of more than an octave.
  - **COMPUTATIONAL NOTE:** A weighted Boolean reward term checking that $$\max\{s^{1}_{t-4},\;\dots,\;s^{1}_{t-1}\}-\min\{s^{1}_{t-4},\;\dots,\;s^{1}_{t-1}\}\;\le\;12.$$
- Avoid consecutive 4ths and 5ths.
  - **COMPUTATIONAL NOTE:** A weighted term that does a check that $$\text{if}\;\;\Delta s^{1}_{t-2}\;=\;\Delta s^{1}_{t-1},\;\;\text{then}\;\;\Delta s^{1}_{t-1}\notin\{5,7\}.$$
### Steps and jumps
- Follow large leaps with *small* step in opposite direction.
    - **COMPUTATIONAL NOTE:** [...]
- Avoid stepwise motion followed by a leap in the opposite direction.
    - **COMPUTATIONAL NOTE:** [...]
### Scale
- Give emphasis to the tonic scale degree.
    - **COMPUTATIONAL NOTE:** [...]
- Shape the melodic line around structural pitches.
  - Put structural pitches at stronger beats
      - **COMPUTATIONAL NOTE:** [...]
  - Avoid putting unstable scale degrees at stronger beats.
      - **COMPUTATIONAL NOTE:** [...]
- In ascending context, resolve the leading tone (7th) to tonic as quickly as possible.
    - **COMPUTATIONAL NOTE:** [...]
### Range
- When moving in one direction, up or down, do not expose unstable intervals such as a $\text{A4}/\text{d5}$ or 7th.
    - **COMPUTATIONAL NOTE:** [...]
- Use V-I cadence at end of passage.
    - **COMPUTATIONAL NOTE:** [...]
## $1^{\text{st}}$-species: simple accompaniment, i.e., a *duo*
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/2nd_species_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="images/2nd_species_light.png">
    <img src="images/2nd_species_light.png" alt="description" width="325" style="opacity: 0.65;">
  </picture>
</p>

### Melodic motion
- Use a variety of parallel, similar, and contrary motion.
  - **COMPUTATIONAL NOTE:** [...]
- Avoid *oblique* motion, that is, motion where one voice moves but the other is stationary.
  - **COMPUTATIONAL NOTE:** [...]
- Avoid "immediate" (how immediate?) repetiion of chords and notes.
  - **COMPUTATIONAL NOTE:** [...]
### Vertical intervals
- Use only consonances for outer intervals in chord (in $1^{\text{st}}$-species, this is whole chord).
  - **COMPUTATIONAL NOTE:** [...]
- Common occurances in vertical intervals:
  - $\text{P8}$ over tonic at beginning or end of phrase
    - **COMPUTATIONAL NOTE:** [...]
  - $\text{P5}$ over dominant pitches at or near beginning or end of phrase.
    - **COMPUTATIONAL NOTE:** [...]
  - $\text{P8}$ can appear over dominant approximately midway through phrase, surrounded by $3$s and $6$s.
    - **COMPUTATIONAL NOTE:** [...]
- Between these loci in the phrase, imperfect cadences should be used to maintina sense of flow.
  - **COMPUTATIONAL NOTE:** [...]
  - Parallel $3$rds and $6$ths are ok, be more than 4 successive is excessive (destroys sense of line indpendence)
    - **COMPUTATIONAL NOTE:** [...]
  - $\text{P5}$ and $\text{P8}$ are ok at these places, but only if surrounded by $\text{3}$ rds and $\text{6}$ ths.
    - **COMPUTATIONAL NOTE:** [...]
- Avoid parallel perfect intevals.
  - **COMPUTATIONAL NOTE:** [...]
### Harmonic succesions
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