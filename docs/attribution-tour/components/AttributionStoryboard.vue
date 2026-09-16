<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

type PhaseId = 'initialize' | 'ingest' | 'extraction' | 'filtering' | 'refinement' | 'collection' | 'result'
type SourceId = 'applog' | 'syslog'
type Severity = 'bad' | 'interesting' | 'context'
type MotionSpeed = 'normal' | 'slow' | 'study'

interface Scene {
  id: string
  phase?: PhaseId
  kicker: string
  title: string
  summary: string
  short: string
  docAnchor: string
}

interface ObservationLine {
  id: string
  line: string
  text: string
  rule?: string
  match?: string
  emits?: string
  event?: string
  severity?: Severity
}

const DOCS_BASE =
  import.meta.env.VITE_ATTRIBUTION_DOCS_BASE ||
  'https://github.com/dsx-ai-factory/attribution/blob/main/docs/attribution.md'

const motionPresets: Record<MotionSpeed, { label: string; title: string; fast: string; medium: string; slow: string }> = {
  normal: { label: '1×', title: 'Normal motion', fast: '180ms', medium: '480ms', slow: '820ms' },
  slow: { label: '½×', title: 'Slow motion', fast: '360ms', medium: '960ms', slow: '1640ms' },
  study: { label: '¼×', title: 'Study motion', fast: '720ms', medium: '1920ms', slow: '3280ms' },
}
const motionOptions: MotionSpeed[] = ['normal', 'slow', 'study']

const phases: Array<{ id: PhaseId; label: string; color: string }> = [
  { id: 'initialize', label: 'Initialize', color: '#475569' },
  { id: 'ingest', label: 'Ingest', color: '#2563eb' },
  { id: 'extraction', label: 'Extraction', color: '#7c3aed' },
  { id: 'filtering', label: 'Filtering', color: '#0f766e' },
  { id: 'refinement', label: 'Refinement', color: '#c2410c' },
  { id: 'collection', label: 'Collection', color: '#a21caf' },
  { id: 'result', label: 'Result', color: '#15803d' },
]

const subphaseGroups: Partial<
  Record<PhaseId, Array<{ id: string; label: string; note: string; effect: 'noop' | 'change' | 'boundary' | 'decision' }>>
> = {
  filtering: [
    { id: 'filter-early', label: 'EARLY', note: '0 fired', effect: 'noop' },
    { id: 'filter-mid', label: 'MID', note: '0 fired', effect: 'noop' },
    { id: 'filter-standard', label: 'STANDARD', note: '1 fired', effect: 'change' },
    { id: 'filter-late', label: 'LATE', note: '0 fired', effect: 'noop' },
  ],
  refinement: [
    { id: 'refine-materialize', label: 'MATERIALIZE', note: 'boundary', effect: 'boundary' },
    { id: 'refine-scrub', label: 'SCRUB', note: '0 fired', effect: 'noop' },
    { id: 'refine-aggregate', label: 'AGGREGATE', note: 'boundary', effect: 'boundary' },
    { id: 'refine-final', label: 'FINAL', note: '2 changes', effect: 'change' },
    { id: 'outcome', label: 'OUTCOME', note: 'selected', effect: 'decision' },
    { id: 'post-outcome', label: 'POST_OUTCOME', note: '1 cleanup', effect: 'change' },
  ],
}

const scenes: Scene[] = [
  {
    id: 'overview',
    kicker: 'NVIDIA Attribution · worked example',
    title: 'Watch evidence become an explanation',
    summary: 'One continuous, inspectable path from completed job information to one root cause.',
    short: 'Overview',
    docAnchor: '#detailed-attribution-phases',
  },
  {
    id: 'initialize',
    phase: 'initialize',
    kicker: 'Initialize',
    title: 'Attribution starts with the completed job',
    summary: 'Before observations arrive, the attribution run knows the job boundary, terminal state, and nodes in scope.',
    short: 'Initialize',
    docAnchor: '#initialization-order',
  },
  {
    id: 'ingest',
    phase: 'ingest',
    kicker: 'Ingest',
    title: 'Two sources add evidence to the job',
    summary: 'Browse the raw records. The same observation pane will move into Extraction.',
    short: 'Ingest',
    docAnchor: '#ingest-attribution-cycle',
  },
  {
    id: 'extraction',
    phase: 'extraction',
    kicker: 'Extraction',
    title: 'A selected line becomes a structured event',
    summary: 'Choose a record to inspect its matcher, extractor, and emitted event.',
    short: 'Extract',
    docAnchor: '#phase-1-extraction',
  },
  {
    id: 'filter-early',
    phase: 'filtering',
    kicker: 'Filtering · EARLY',
    title: 'The stream enters ordered filter strata',
    summary: 'Every stratum reads the stream state left by the preceding one.',
    short: 'Early',
    docAnchor: '#phase-2-filtering',
  },
  {
    id: 'filter-mid',
    phase: 'filtering',
    kicker: 'Filtering · MID',
    title: 'Unmatched rules leave the stream intact',
    summary: 'No MID rule applies to this evidence, so every event keeps its state.',
    short: 'Mid',
    docAnchor: '#filter',
  },
  {
    id: 'filter-standard',
    phase: 'filtering',
    kicker: 'Filtering · STANDARD',
    title: 'Two events establish the meaning of one',
    summary: 'A later MFU marker changes how an earlier CUDA error is interpreted.',
    short: 'Standard',
    docAnchor: '#filter',
  },
  {
    id: 'filter-late',
    phase: 'filtering',
    kicker: 'Filtering · LATE',
    title: 'The transformed stream leaves Filtering',
    summary: 'The demotion persists; unaffected events continue in their original order.',
    short: 'Late',
    docAnchor: '#phase-2-filtering',
  },
  {
    id: 'refine-materialize',
    phase: 'refinement',
    kicker: 'Refinement · materialize',
    title: 'Events become per-node attribution groups',
    summary: 'This framework boundary groups filtered evidence by the node it describes.',
    short: 'By node',
    docAnchor: '#phase-3-refinement',
  },
  {
    id: 'refine-scrub',
    phase: 'refinement',
    kicker: 'Refinement · SCRUB',
    title: 'SCRUB rules run independently for each node',
    summary: 'Each node gets the same rule pass. No SCRUB rule matches either node in this example.',
    short: 'Scrub',
    docAnchor: '#refiner',
  },
  {
    id: 'refine-aggregate',
    phase: 'refinement',
    kicker: 'Refinement · aggregate',
    title: 'Node symptoms form one job-wide set',
    summary: 'Outcome-driving symptoms move into a global set while their node provenance stays attached.',
    short: 'Aggregate',
    docAnchor: '#phase-3-refinement',
  },
  {
    id: 'refine-final',
    phase: 'refinement',
    kicker: 'Refinement · FINAL',
    title: 'FINAL rules reason across the whole job',
    summary: 'The stronger kernel failure explains the generic oops; job state adds a fallback symptom.',
    short: 'Final',
    docAnchor: '#refiner',
  },
  {
    id: 'outcome',
    phase: 'refinement',
    kicker: 'Refinement · outcome mapping',
    title: 'Mappings rank the remaining explanations',
    summary: 'The most specific, highest-priority symptom supplies the provisional outcome.',
    short: 'Outcome',
    docAnchor: '#outcome-mapping',
  },
  {
    id: 'post-outcome',
    phase: 'refinement',
    kicker: 'Refinement · POST_OUTCOME',
    title: 'Outcome-aware rules get both sides of the decision',
    summary: 'The provisional outcome and global symptoms are refined together into final state.',
    short: 'Post',
    docAnchor: '#refiner',
  },
  {
    id: 'collection',
    phase: 'collection',
    kicker: 'Collection',
    title: 'Collectors read the stream without changing it',
    summary: 'A collector selects the version event and formats it for the result.',
    short: 'Collect',
    docAnchor: '#phase-4-collections',
  },
  {
    id: 'result',
    phase: 'result',
    kicker: 'Attribution result',
    title: 'One result preserves the complete reasoning chain',
    summary: 'Outcome, root cause, supporting evidence, collectors, and provenance remain inspectable.',
    short: 'Result',
    docAnchor: '#phase-5-reporting',
  },
]

const observations: Record<SourceId, { label: string; node: string; lines: ObservationLine[] }> = {
  applog: {
    label: 'APPLOG',
    node: 'node-01',
    lines: [
      { id: 'boot', line: '0126', text: 'initializing distributed training' },
      {
        id: 'cuda',
        line: '0127',
        text: '0: CUDA error: an illegal memory access was encountered',
        match: 'an illegal memory access was encountered',
        rule: 'CudaError',
        emits: 'Attribution.node.bad',
        event: 'cudaErrorIllegalAddress',
        severity: 'bad',
      },
      {
        id: 'mfu',
        line: '0128',
        text: '0: [16:27:00] iteration 516520 · 399.3 TFLOP/s/GPU',
        match: 'iteration … throughput per GPU',
        rule: 'MFU',
        emits: 'Application.mfu',
        event: 'MFU progress marker',
        severity: 'context',
      },
      {
        id: 'version',
        line: '0129',
        text: '0: NCCL version 2.18.5+cuda12.2',
        match: 'NCCL version 2.18.5+cuda12.2',
        rule: 'NCCLCudaVersion',
        emits: 'Application.nccl.version',
        event: 'NCCL / CUDA version',
        severity: 'context',
      },
      { id: 'checkpoint', line: '0130', text: 'checkpoint writer stopped' },
    ],
  },
  syslog: {
    label: 'SYSLOG',
    node: 'node-02',
    lines: [
      { id: 'heartbeat', line: '8841', text: 'kernel: node heartbeat nominal' },
      {
        id: 'kernel',
        line: '8842',
        text: 'Unable to handle kernel NULL pointer dereference at 0008',
        match: 'Unable to handle kernel NULL pointer dereference',
        rule: 'SyslogKernelInvalidMemoryAccess',
        emits: 'Attribution.node.bad',
        event: 'kernel invalid memory access',
        severity: 'bad',
      },
      {
        id: 'oops',
        line: '8843',
        text: 'Internal error: Oops: 96000007 [#1] SMP',
        match: 'Internal error: Oops',
        rule: 'SyslogKernelOops',
        emits: 'Attribution.node.bad',
        event: 'kernel oops',
        severity: 'bad',
      },
      { id: 'trace', line: '8844', text: 'Call trace: dump_backtrace+0x0/0x1a8' },
    ],
  },
}

const events = [
  { id: 'cuda', label: 'CUDA illegal address', node: 'node-01', time: '16:26:50', kind: 'symptom' },
  { id: 'mfu', label: 'MFU progress marker', node: 'node-01', time: '16:27:00', kind: 'context' },
  { id: 'version', label: 'NCCL 2.18.5 · CUDA 12.2', node: 'node-01', time: '16:27:02', kind: 'context' },
  { id: 'kernel', label: 'kernel invalid memory', node: 'node-02', time: '16:27:30', kind: 'symptom' },
  { id: 'oops', label: 'kernel oops', node: 'node-02', time: '16:27:30', kind: 'symptom' },
]

const currentIndex = ref(0)
const selectedSource = ref<SourceId>('applog')
const selectedLineId = ref('cuda')
const motionSpeed = ref<MotionSpeed>('slow')

const current = computed(() => scenes[currentIndex.value])
const currentPhaseIndex = computed(() => phases.findIndex((phase) => phase.id === current.value.phase))
const currentObservation = computed(() => observations[selectedSource.value])
const selectedLine = computed(
  () => currentObservation.value.lines.find((line) => line.id === selectedLineId.value) ?? currentObservation.value.lines[0],
)
const docsHref = computed(() => `${DOCS_BASE}${current.value.docAnchor}`)
const isFiltering = computed(() => current.value.id.startsWith('filter-'))
const currentSubphases = computed(() => (current.value.phase ? subphaseGroups[current.value.phase] ?? [] : []))
const filterPriority = computed(() => (isFiltering.value ? current.value.id.replace('filter-', '').toUpperCase() : ''))
const refineStep = computed(() => current.value.id.replace('refine-', ''))
const motionStyle = computed(() => {
  const preset = motionPresets[motionSpeed.value]
  return {
    '--motion-fast': preset.fast,
    '--motion-medium': preset.medium,
    '--motion-slow': preset.slow,
  }
})

function sceneIs(id: string) {
  return current.value.id === id
}

function sceneAtOrAfter(id: string) {
  return currentIndex.value >= scenes.findIndex((scene) => scene.id === id)
}

function phaseState(index: number) {
  if (currentPhaseIndex.value < 0) return 'upcoming'
  if (index < currentPhaseIndex.value) return 'complete'
  if (index === currentPhaseIndex.value) return 'active'
  return 'upcoming'
}

function subphaseState(index: number) {
  const active = currentSubphases.value.findIndex((subphase) => subphase.id === current.value.id)
  if (index < active) return 'complete'
  if (index === active) return 'active'
  return 'upcoming'
}

function eventSeverity(id: string): Severity {
  if (id === 'mfu' || id === 'version') return 'context'
  if (id === 'cuda' && sceneAtOrAfter('filter-standard')) return 'interesting'
  return 'bad'
}

function selectSource(source: SourceId) {
  selectedSource.value = source
  selectedLineId.value = observations[source].lines.find((line) => line.rule)?.id ?? observations[source].lines[0].id
}

function goTo(index: number) {
  currentIndex.value = Math.max(0, Math.min(scenes.length - 1, index))
  if (typeof window !== 'undefined') {
    const url = new URL(window.location.href)
    url.searchParams.set('scene', current.value.id)
    window.history.replaceState({}, '', url)
  }
}

function next() {
  goTo(currentIndex.value + 1)
}

function previous() {
  goTo(currentIndex.value - 1)
}

function setMotionSpeed(speed: MotionSpeed) {
  motionSpeed.value = speed
  if (typeof window !== 'undefined') {
    const url = new URL(window.location.href)
    url.searchParams.set('motion', speed)
    window.history.replaceState({}, '', url)
  }
}

function onKeydown(event: KeyboardEvent) {
  const target = event.target as HTMLElement | null
  if (target?.closest('button, a, input, textarea, select')) return
  if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {
    event.preventDefault()
    event.stopImmediatePropagation()
    next()
  } else if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
    event.preventDefault()
    event.stopImmediatePropagation()
    previous()
  } else if (event.key === 'Home') {
    event.preventDefault()
    event.stopImmediatePropagation()
    goTo(0)
  } else if (event.key === 'End') {
    event.preventDefault()
    event.stopImmediatePropagation()
    goTo(scenes.length - 1)
  }
}

onMounted(() => {
  const url = new URL(window.location.href)
  const requestedScene = url.searchParams.get('scene')
  const requested = requestedScene === 'observations' ? 'ingest' : requestedScene
  const index = scenes.findIndex((scene) => scene.id === requested)
  if (index >= 0) currentIndex.value = index
  const requestedMotion = url.searchParams.get('motion')
  if (requestedMotion && motionOptions.includes(requestedMotion as MotionSpeed)) {
    motionSpeed.value = requestedMotion as MotionSpeed
  }
  window.addEventListener('keydown', onKeydown, true)
})

onBeforeUnmount(() => window.removeEventListener('keydown', onKeydown, true))
</script>

<template>
  <div class="storyboard" :data-scene="current.id" :data-phase="current.phase ?? 'overview'" :style="motionStyle">
    <nav class="major-rail" aria-label="Attribution phases">
      <button
        v-for="(phase, index) in phases"
        :key="phase.id"
        type="button"
        class="major-phase"
        :class="phaseState(index)"
        :style="{ '--phase-color': phase.color }"
        :disabled="phaseState(index) === 'upcoming'"
        @click="goTo(scenes.findIndex((scene) => scene.phase === phase.id))"
      >
        <span class="major-marker" />
        <span>{{ phase.label }}</span>
      </button>
    </nav>

    <nav
      v-if="currentSubphases.length"
      class="subphase-rail"
      :class="current.phase"
      :aria-label="`${current.phase} sub-phases`"
    >
      <button
        v-for="(subphase, index) in currentSubphases"
        :key="subphase.id"
        type="button"
        :class="[subphaseState(index), `effect-${subphase.effect}`]"
        :aria-current="subphase.id === current.id ? 'step' : undefined"
        :aria-label="`${subphase.label}: ${subphase.note}`"
        @click="goTo(scenes.findIndex((scene) => scene.id === subphase.id))"
      >
        <i /><span>{{ subphase.label }}</span><small>{{ subphase.note }}</small>
      </button>
    </nav>

    <a class="docs-link" :href="docsHref" target="_blank" rel="noreferrer">
      Read the canonical process documentation <span>↗</span>
    </a>

    <header v-if="!sceneIs('overview')" class="story-header">
      <div class="story-kicker">{{ current.kicker }}</div>
      <h1>{{ current.title }}</h1>
      <p>{{ current.summary }}</p>
    </header>

    <main class="story-canvas">
      <section class="overview-scene" :class="{ visible: sceneIs('overview') }">
        <div class="overview-eyebrow">NVIDIA Attribution · interactive walkthrough</div>
        <h1>Evidence becomes<br><span>an explanation</span></h1>
        <p>Follow the same job context, observations, events, and symptoms as they change shape across the pipeline.</p>
        <div class="overview-chain">
          <span>1 completed job</span><i>→</i><span>2 observations</span><i>→</i><span>5 events</span><i>→</i><span>2 nodes</span><i>→</i><b>1 root cause</b>
        </div>
        <button type="button" class="begin-button" @click="next">Begin with completed job information <span>→</span></button>
      </section>

      <section class="initialization-stage" :class="{ visible: sceneIs('initialize') }">
        <article class="job-info-card">
          <header>
            <div><span>COMPLETED JOB INFORMATION</span><strong>walkthrough-42</strong></div>
            <b>immutable context</b>
          </header>
          <dl>
            <div class="job-state"><dt>terminal state</dt><dd>FAILED</dd></div>
            <div><dt>started</dt><dd>2025-04-15 · 16:26:40Z</dd></div>
            <div><dt>ended</dt><dd>2025-04-15 · 16:28:20Z</dd></div>
          </dl>
          <div class="job-nodes">
            <span>NODELIST</span><b>node-01</b><b>node-02</b>
          </div>
        </article>
        <div class="initialization-arrow"><span>initialize once</span><i>→</i></div>
        <article class="run-card">
          <div class="run-mark">A</div>
          <span>ATTRIBUTION RUN</span>
          <strong>Initialized</strong>
          <p>Job scope and terminal state are now available to every later phase.</p>
          <div><i />ready for observations</div>
        </article>
      </section>

      <section
        class="persistent-observation"
        :class="{
          centered: sceneIs('ingest'),
          docked: sceneIs('extraction'),
          offstage: !(sceneIs('ingest') || sceneIs('extraction')),
        }"
        aria-label="Source observation"
      >
        <div class="observation-toolbar">
          <div>
            <strong>Observation</strong>
            <span>job walkthrough-42</span>
          </div>
          <span class="observation-count">2 source observations</span>
        </div>
        <nav class="source-tabs" aria-label="Source observations" role="tablist">
          <button
            v-for="source in (['applog', 'syslog'] as SourceId[])"
            :key="source"
            type="button"
            role="tab"
            :aria-selected="selectedSource === source"
            :class="{ active: selectedSource === source }"
            @click="selectSource(source)"
          >
            <i aria-hidden="true" />
            <span><strong>{{ observations[source].label }}</strong><small>{{ observations[source].node }}</small></span>
            <b v-if="sceneIs('extraction')">{{ observations[source].lines.filter((line) => line.rule).length }} matches</b>
            <b v-else>{{ observations[source].lines.length }} lines</b>
          </button>
        </nav>
        <div class="observation-meta">
          <span>{{ currentObservation.node }}</span><span>{{ currentObservation.lines.length }} source records</span><span>ingested</span>
        </div>
        <div class="observation-lines">
          <button
            v-for="line in currentObservation.lines"
            :key="line.id"
            type="button"
            class="observation-line"
            :class="{ selected: selectedLineId === line.id, matched: line.rule && sceneIs('extraction') }"
            @click="selectedLineId = line.id"
          >
            <span>{{ line.line }}</span><code>{{ line.text }}</code><i v-if="line.rule && sceneIs('extraction')">match</i>
          </button>
        </div>
        <div class="scroll-cue"><span>scrollable observation</span><i>↕</i></div>
      </section>

      <section class="extractor-card" :class="{ visible: sceneIs('extraction') }">
        <div class="connector-label">selected record</div>
        <div class="extractor-heading"><span>Extractor rule</span><strong>{{ selectedLine.rule ?? 'No match' }}</strong></div>
        <dl v-if="selectedLine.rule">
          <div><dt>matches</dt><dd>{{ selectedLine.match }}</dd></div>
          <div><dt>emits</dt><dd>{{ selectedLine.emits }}</dd></div>
        </dl>
        <div v-else class="no-match">This line passes without emitting an event.</div>
        <div v-if="selectedLine.event" class="emission" :class="selectedLine.severity">
          <span>{{ selectedLine.severity }}</span><strong>{{ selectedLine.event }}</strong>
        </div>
      </section>

      <section
        class="persistent-stream"
        :class="{
          extraction: sceneIs('extraction'),
          filtering: isFiltering,
          materializing: current.id.startsWith('refine-'),
          collecting: sceneIs('collection'),
          hidden: !(sceneIs('extraction') || isFiltering || sceneIs('refine-materialize') || sceneIs('collection')),
        }"
        aria-label="Job event stream"
      >
        <div class="stream-heading"><span>Event stream</span><small>ordered · job scoped</small></div>
        <div class="stream-axis"><span>earlier</span><i /><span>later</span></div>
        <article
          v-for="event in events"
          :key="event.id"
          class="stream-event"
          :class="[
            eventSeverity(event.id),
            { 'symptom-event': event.kind === 'symptom', 'context-event': event.kind === 'context' },
            { selected: event.id === selectedLineId && sceneIs('extraction') },
          ]"
        >
          <div><span>{{ event.time }}</span><small>{{ event.node }}</small></div>
          <strong>{{ event.label }}</strong>
          <b>{{ eventSeverity(event.id) }}</b>
        </article>
      </section>

      <section class="filter-stage" :class="{ visible: isFiltering }">
        <div class="filter-rule-card" :class="{ relational: sceneIs('filter-standard') }">
          <template v-if="sceneIs('filter-standard')">
            <span class="rule-band">STANDARD rule</span>
            <h3>DemoteTflopsSymptoms</h3>
            <p><b>Target:</b> eligible CUDA error at 16:26:50</p>
            <p><b>Context:</b> last MFU marker at 16:27:00</p>
            <div class="predicate"><code>error.time &lt; last_mfu.time</code><span>matched</span></div>
          </template>
          <template v-else-if="sceneIs('filter-late')">
            <span class="rule-band">LATE rules</span>
            <h3>0 rules fired</h3>
            <p>The earlier demotion remains part of the current stream state.</p>
            <div class="unchanged-badge">ready for materialization →</div>
          </template>
          <template v-else>
            <span class="rule-band">{{ filterPriority }} rules</span>
            <h3>0 rules fired</h3>
            <p>The complete stream advances unchanged to the next priority.</p>
            <div class="unchanged-badge">5 events → 5 events</div>
          </template>
        </div>

        <svg v-if="sceneIs('filter-standard')" class="filter-connectors" viewBox="0 0 1120 330" aria-hidden="true">
          <path d="M116 118 C160 205, 320 205, 450 253" />
          <path d="M338 118 C365 188, 420 204, 510 253" />
          <circle cx="116" cy="118" r="5" /><circle cx="338" cy="118" r="5" />
        </svg>
      </section>

      <section
        class="refiner-stage"
        :class="{ visible: current.id.startsWith('refine-') }"
        :data-refine="refineStep"
      >
        <div class="scope-band node-scope"><span>PER-NODE STATE</span><small>materialize · SCRUB</small></div>
        <div class="scope-band global-scope"><span>JOB-WIDE STATE</span><small>aggregate · FINAL</small></div>

        <div class="node-shell node-one">
          <header><span>N1</span><div><strong>node-01</strong><small>APPLOG</small></div></header>
          <div v-if="sceneIs('refine-scrub')" class="scrub-stamp"><b>SCRUB</b><span>0 matches · unchanged</span></div>
        </div>
        <div class="node-shell node-two">
          <header><span>N2</span><div><strong>node-02</strong><small>SYSLOG</small></div></header>
          <div v-if="sceneIs('refine-scrub')" class="scrub-stamp"><b>SCRUB</b><span>0 matches · unchanged</span></div>
        </div>
        <div class="global-shell">
          <header><span>GLOBAL SYMPTOM SET</span><small>all nodes · provenance retained</small></header>
        </div>
        <div class="evidence-shelf"><span>SUPPORTING EVIDENCE</span></div>

        <article class="symptom-card symptom-cuda interesting">
          <small>node-01</small><strong>cudaErrorIllegalAddress</strong><b>interesting</b>
        </article>
        <article class="symptom-card symptom-kernel bad">
          <small>node-02</small><strong>kernel invalid memory access</strong><b>bad</b>
        </article>
        <article class="symptom-card symptom-oops" :class="sceneAtOrAfter('refine-final') ? 'interesting' : 'bad'">
          <small>node-02</small><strong>kernel oops</strong><b>{{ sceneAtOrAfter('refine-final') ? 'interesting' : 'bad' }}</b>
        </article>
        <article class="symptom-card symptom-job bad">
          <small>job state</small><strong>job failed</strong><b>bad</b>
        </article>

        <div v-if="sceneIs('refine-materialize')" class="boundary-note materialize-note">
          <b>framework boundary</b><span>symptoms → node groups · context remains stream-only</span>
        </div>
        <div v-if="sceneIs('refine-scrub')" class="boundary-note scrub-note">
          <b>0 rules fired</b><span>each node group passes through unchanged</span>
        </div>
        <div v-if="sceneIs('refine-aggregate')" class="boundary-note aggregate-note">
          <b>aggregation boundary</b><span>remaining bad symptoms become one job-wide set</span>
        </div>
        <div v-if="sceneIs('refine-final')" class="final-rule-card">
          <span>FINAL · DAG cascade</span>
          <div><code>kernel invalid memory access</code><i>explains</i><code>kernel oops</code></div>
          <small>demote the subordinate symptom; add job-state fallback</small>
        </div>
      </section>

      <section class="outcome-stage" :class="{ visible: sceneIs('outcome') }">
        <div class="outcome-input">
          <span>GLOBAL SYMPTOMS</span>
          <div><b class="bad-dot" />kernel invalid memory access<small>node-02</small></div>
          <div><b class="bad-dot" />job failed<small>job state</small></div>
        </div>
        <div class="mapping-arrow"><span>rank mappings</span><i>→</i></div>
        <div class="mapping-candidates">
          <article class="mapping-card selected">
            <header><span>MEDIUM_SW</span><b>selected</b></header>
            <code>kernel invalid memory access</code>
            <strong>Fail · InfraSW</strong>
            <small>specific kernel mapping</small>
          </article>
          <article class="mapping-card">
            <header><span>FALLBACK</span></header>
            <code>job failed</code>
            <strong>Fail · User</strong>
            <small>generic job-state mapping</small>
          </article>
        </div>
      </section>

      <section class="post-stage" :class="{ visible: sceneIs('post-outcome') }">
        <div class="post-input symptoms-input">
          <span>SYMPTOMS</span><b>kernel invalid memory</b><b>job failed</b>
        </div>
        <div class="plus">+</div>
        <div class="post-input outcome-input-card">
          <span>PROVISIONAL OUTCOME</span><b>Fail · InfraSW</b><small>kernel invalid memory access</small>
        </div>
        <div class="post-arrow">→</div>
        <div class="post-rule">
          <span>POST_OUTCOME</span><strong>PostProcessSymptoms</strong><code>remove “job failed”</code>
        </div>
        <div class="post-arrow">→</div>
        <div class="post-output">
          <span>FINAL STATE</span><strong>Fail · InfraSW</strong><b>kernel invalid memory access</b>
        </div>
      </section>

      <section class="collector-stage" :class="{ visible: sceneIs('collection') }">
        <div class="collector-connector stream-to-collector" aria-hidden="true"><span>select</span></div>
        <div class="collector-connector collector-to-result" aria-hidden="true"><span>format</span></div>
        <article class="collector-rule">
          <span>COLLECTOR RULE</span><strong>Collect.NCCLCudaVersion</strong>
          <p>Select the first <code>Application.nccl.version</code> event.</p>
          <b>read only</b>
        </article>
        <article class="collector-output">
          <span>COLLECTOR RESULT</span>
          <code>{<br>&nbsp; nccl_version: "2.18.5",<br>&nbsp; cuda_version: "12.2"<br>}</code>
        </article>
      </section>

      <section class="result-stage" :class="{ visible: sceneIs('result') }">
        <article class="final-outcome">
          <span>ROOT CAUSE</span><h2>Fail</h2><strong>InfraSW</strong><code>kernel invalid memory access</code>
        </article>
        <div class="result-column">
          <article><span>SUPPORTING EVIDENCE</span><b>cudaErrorIllegalAddress</b><b>kernel oops</b></article>
          <article><span>COLLECTORS</span><b>NCCL 2.18.5</b><b>CUDA 12.2</b></article>
        </div>
        <div class="provenance-chain">
          <span>observation</span><i>→</i><span>extractor</span><i>→</i><span>event</span><i>→</i><span>refiner</span><i>→</i><b>outcome</b>
        </div>
      </section>
    </main>

    <nav class="story-controls" aria-label="Storyboard navigation">
      <button type="button" class="previous" :disabled="currentIndex === 0" @click="previous"><span>←</span> Previous</button>
      <div class="scene-progress">
        <button
          v-for="(scene, index) in scenes"
          :key="scene.id"
          type="button"
          :class="{ active: index === currentIndex, complete: index < currentIndex }"
          :title="scene.short"
          :aria-label="`Open ${scene.short}`"
          @click="goTo(index)"
        />
        <span>{{ current.short }}</span>
        <div class="motion-control" role="group" aria-label="Animation speed">
          <span>Motion</span>
          <button
            v-for="speed in motionOptions"
            :key="speed"
            type="button"
            :class="{ active: motionSpeed === speed }"
            :title="motionPresets[speed].title"
            :aria-pressed="motionSpeed === speed"
            @click="setMotionSpeed(speed)"
          >
            {{ motionPresets[speed].label }}
          </button>
        </div>
      </div>
      <button type="button" class="next" :disabled="currentIndex === scenes.length - 1" @click="next">
        Next <span>→</span>
      </button>
    </nav>
  </div>
</template>
