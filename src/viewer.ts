import {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Mesh,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SphereGeometry,
  TorusGeometry,
  CylinderGeometry,
  WebGPURenderer,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
  MeshPhysicalNodeMaterial,
} from 'three/webgpu'
import {
  DataTexture,
  NoToneMapping,
  RGBAFormat,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  UnsignedByteType,
  Vector2,
} from 'three'
import WebGPU from 'three/addons/capabilities/WebGPU.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import * as TSL from 'three/tsl'
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js'

const container = document.getElementById('viewer')
if (!container) {
  throw new Error('Viewer container not found')
}

const textureLoader = new TextureLoader()
const textureSources: Record<string, { src: string; name?: string }> = {}
const textureSignatures: Record<string, string> = {}
const textureCache = new Map<string, Texture>()
const ktx2Loader = new KTX2Loader().setTranscoderPath('/basis/')
let ktx2Ready = false
let ktx2InitPromise: Promise<void> | null = null
const fallbackTexture = new DataTexture(
  new Uint8Array([255, 255, 255, 255]),
  1,
  1,
  RGBAFormat,
  UnsignedByteType,
)
fallbackTexture.colorSpace = SRGBColorSpace
fallbackTexture.needsUpdate = true

const timeUniform = TSL.uniform(0)

const ensureKtx2Support = async () => {
  if (ktx2Ready) return
  if (ktx2InitPromise) return ktx2InitPromise
  if (!WebGPU.isAvailable()) return
  ktx2InitPromise = (async () => {
    const renderer = new WebGPURenderer()
    await renderer.init()
    ktx2Loader.detectSupport(renderer)
    renderer.dispose()
    ktx2Ready = true
  })().finally(() => {
    ktx2InitPromise = null
  })
  return ktx2InitPromise
}

const isKtx2Entry = (entry?: { src: string; name?: string }) =>
  Boolean(entry?.name?.toLowerCase().endsWith('.ktx2')) ||
  Boolean(entry?.src?.toLowerCase().includes('.ktx2'))

const textureFromNode = (id: string) => {
  const entry = textureSources[id]
  const src = entry?.src ?? ''
  if (!src) return fallbackTexture
  const previous = textureSignatures[id]
  const cached = textureCache.get(id)
  if (previous && previous === src && cached) {
    return cached
  }
  const placeholder = cached ?? new Texture()
  placeholder.colorSpace = SRGBColorSpace
  textureCache.set(id, placeholder)
  textureSignatures[id] = src
  if (isKtx2Entry(entry)) {
    void ensureKtx2Support()?.then(() => {
      ktx2Loader.load(
        src,
        (loaded) => {
          placeholder.copy(loaded)
          placeholder.colorSpace = SRGBColorSpace
          placeholder.needsUpdate = true
          if (lastCode) {
            applyCode(lastCode, lastTextures, lastGeometry)
          }
        },
        undefined,
        () => {},
      )
    })
    return placeholder
  }
  textureLoader.load(
    src,
    (loaded) => {
      placeholder.copy(loaded)
      placeholder.colorSpace = SRGBColorSpace
      placeholder.needsUpdate = true
      if (lastCode) {
        applyCode(lastCode, lastTextures, lastGeometry)
      }
    },
    undefined,
    () => {},
  )
  return placeholder
}

const runtime = {
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Mesh,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SphereGeometry,
  TorusGeometry,
  CylinderGeometry,
  WebGPURenderer,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
  MeshPhysicalNodeMaterial,
  SRGBColorSpace,
  NoToneMapping,
  Vector2,
  OrbitControls,
  WebGPU,
  TSL,
}

let lastCode = ''
let lastTextures: Record<string, { src: string; name?: string }> = {}
let lastGeometry = 'box'
let currentApp: { dispose?: () => void } | null = null

const applyCode = (
  code: string,
  textures: Record<string, { src: string; name?: string }>,
  geometryType: string,
) => {
  lastCode = code
  lastTextures = textures
  lastGeometry = geometryType
  Object.keys(textureSources).forEach((key) => {
    if (!textures[key]) {
      textureSources[key] = { src: '' }
    }
  })
  Object.entries(textures).forEach(([id, payload]) => {
    textureSources[id] = payload
  })
  Object.entries(textures).forEach(([id]) => {
    textureFromNode(id)
  })

  try {
    const fn = new Function(
      'runtime',
      `${code}`,
    ) as (runtimeValue: typeof runtime) =>
      | ((options: {
          container: HTMLElement
          textures?: Record<string, Texture>
          timeUniform?: ReturnType<typeof TSL.uniform>
          geometryType?: string
        }) => { dispose?: () => void })
      | undefined

    const createApp = fn(runtime)
    if (typeof createApp !== 'function') {
      return
    }

    if (currentApp?.dispose) {
      currentApp.dispose()
    }
    container.innerHTML = ''

    const textureMap: Record<string, Texture> = {}
    Object.keys(textureSources).forEach((id) => {
      textureMap[id] = textureFromNode(id)
    })

    currentApp = createApp({
      container,
      textures: textureMap,
      timeUniform,
      geometryType,
    })
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Unknown viewer error'
    window.parent?.postMessage({ type: 'tsl-viewer-error', message }, '*')
  }
}

window.addEventListener('message', (event) => {
  const data = event.data as
    | {
        type?: string
        code?: string
        textures?: Record<string, { src: string; name?: string }>
        geometryType?: string
      }
    | undefined
  if (data?.type === 'tsl-code' && typeof data.code === 'string') {
    applyCode(data.code, data.textures ?? {}, data.geometryType ?? 'box')
  }
})

window.parent?.postMessage({ type: 'tsl-viewer-ready' }, '*')
