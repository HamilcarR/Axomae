#include "../includes/Drawable.h"

using namespace axomae;

Drawable::Drawable() {
  mesh_object = nullptr;
  camera_pointer = nullptr;
}

Drawable::Drawable(Mesh &mesh) {
  mesh_object = new Mesh(mesh);
  camera_pointer = nullptr;
  gl_buffers.setGeometryPointer(&mesh_object->geometry);
  initialize();
}

Drawable::Drawable(Mesh *mesh) {
  assert(mesh != nullptr);
  if (mesh != nullptr) {
    mesh_object = mesh;
    camera_pointer = nullptr;
    gl_buffers.setGeometryPointer(&mesh_object->geometry);
    if (!initialize())
      LOG("A problem prevented mesh initialization!", LogLevel::ERROR);
  }
}

Drawable::~Drawable() {}

void Drawable::clean() {
  gl_buffers.clean();
  mesh_object->clean();
  delete mesh_object;
}

bool Drawable::ready() {
  return gl_buffers.isReady();
}

bool Drawable::initialize() {
  if (mesh_object == nullptr)
    return false;
  mesh_object->initializeGlData();
  errorCheck(__FILE__, __LINE__);
  gl_buffers.initializeBuffers();
  errorCheck(__FILE__, __LINE__);
  return gl_buffers.isReady();
  ;
}

void Drawable::startDraw() {
  if (mesh_object != nullptr) {
    mesh_object->bindShaders();
    errorCheck(__FILE__, __LINE__);
    gl_buffers.bindVao();
    gl_buffers.fillBuffers();

    gl_buffers.bindVertexBuffer();
    mesh_object->getShader()->enableAttributeArray(0);
    mesh_object->getShader()->setAttributeBuffer(0, GL_FLOAT, 0, 3, 0);

    gl_buffers.bindColorBuffer();
    mesh_object->getShader()->enableAttributeArray(1);
    mesh_object->getShader()->setAttributeBuffer(1, GL_FLOAT, 0, 3, 0);

    gl_buffers.bindNormalBuffer();
    mesh_object->getShader()->enableAttributeArray(2);
    mesh_object->getShader()->setAttributeBuffer(2, GL_FLOAT, 0, 3, 0);

    gl_buffers.bindTextureBuffer();
    mesh_object->getShader()->enableAttributeArray(3);
    mesh_object->getShader()->setAttributeBuffer(3, GL_FLOAT, 0, 2, 0);

    gl_buffers.bindTangentBuffer();
    mesh_object->getShader()->enableAttributeArray(4);
    mesh_object->getShader()->setAttributeBuffer(4, GL_FLOAT, 0, 3, 0);

    gl_buffers.unbindVao();
    mesh_object->releaseShaders();
  }
}

void Drawable::bind() {
  gl_buffers.bind();
  mesh_object->setupAndBind();
  mesh_object->bindMaterials();
}

void Drawable::unbind() {
  gl_buffers.unbind();
  mesh_object->unbindMaterials();
  mesh_object->afterRenderSetup();
  mesh_object->releaseShaders();
}

void Drawable::setSceneCameraPointer(Camera *camera) {
  camera_pointer = camera;
  mesh_object->setSceneCameraPointer(camera);
}

Shader *Drawable::getMeshShaderPointer() const {
  if (mesh_object)
    return mesh_object->getShader();
  else
    return nullptr;
}

Material *Drawable::getMaterialPointer() const {
  if (mesh_object)
    return mesh_object->getMaterial();
  else
    return nullptr;
}
