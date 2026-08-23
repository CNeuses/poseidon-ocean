export function releaseStorageAttribute(renderer, nodeOrAttribute) {
  const attribute = nodeOrAttribute?.value ?? nodeOrAttribute;
  if (!attribute) return;

  // Three r185 does not expose BufferAttribute.dispose(). The renderer's
  // attribute registry is the single owner of the GPUBuffer and is therefore
  // the matching release boundary for package-owned compute attributes.
  renderer?._attributes?.delete(attribute);
}
