namespace rl {

void RainbowColor(float hue, unsigned char &r, unsigned char &g, unsigned char &b) {
    if(hue <= 0.0)
        hue = 0.000001;
    if(hue >= 1.0)
        hue = 0.999999;

    int h = int(hue * 256 * 6);
    int x = h % 0x100;

    switch (h / 256)
    {
    case 0: r = 255; g = x;       break;
    case 1: g = 255; r = 255 - x; break;
    case 2: g = 255; b = x;       break;
    case 3: b = 255; g = 255 - x; break;
    case 4: b = 255; r = x;       break;
    case 5: r = 255; b = 255 - x; break;
    }
}

Mesh GenMeshCloudPoint(float *pointsArr, int size){
    const int MAX_MESH_VBO = 7;

    // in the points must be a  X, Y, Z   for each points
    Mesh cloudMesh = {0};
    cloudMesh.vboId = (unsigned int *)RL_CALLOC(MAX_MESH_VBO, sizeof(unsigned int));
    int points = size; 
    // Creating a Mesh with only points (vertices) with normals to z=1
    int rest = points % 3;
    // i need to be safe for the index generation of a mesh!
    if(rest != 0)
        points -= rest;
    float *vertices = new float[points*3];
    float *normals  = new float[points*3];
    unsigned char *colors   = new unsigned char[points*4];
    for(int i = 0; i < points; i++)
    {
        vertices[i*3+0] = pointsArr[i*3 + 0]; //x
        vertices[i*3+1] = pointsArr[i*3 + 1]; //y
        vertices[i*3+2] = pointsArr[i*3 + 2]; //z

        normals[i*3+0] = 0;
        normals[i*3+1] = 1;
        normals[i*3+2] = 0;

        // color points by z height, 60mt is space for a comple rainbow
        float zmod = 20;  
        float hue = fmod(fabs(pointsArr[i*3 + 2]),zmod) / zmod;
        unsigned char r,g,b;
        RainbowColor(hue, r, g, b);
        colors[i*4+0] = r;
        colors[i*4+1] = g;
        colors[i*4+2] = b;
        colors[i*4+3] = 255;
    }
    // TexCoords definition
    cloudMesh.texcoords = (float *)RL_MALLOC(points*2*sizeof(float));
    for (int i = 0; i < points; i++)
    {
        cloudMesh.texcoords[2*i] = 0;
        cloudMesh.texcoords[2*i + 1] = 0;
    }

    cloudMesh.vertices = (float *)RL_MALLOC(points*3*sizeof(float));
    memcpy(cloudMesh.vertices, vertices, points*3*sizeof(float));

    cloudMesh.normals = (float *)RL_MALLOC(points*3*sizeof(float));
    memcpy(cloudMesh.normals, normals, points*3*sizeof(float));

    cloudMesh.colors = (unsigned char *)RL_MALLOC(points*4*sizeof(unsigned char));
    memcpy(cloudMesh.colors, colors, points*4*sizeof(unsigned char));

    cloudMesh.vertexCount = points;
    cloudMesh.triangleCount = points/3;

    rlLoadMesh(&cloudMesh, false);

    delete [] vertices;
    delete [] normals;
    delete [] colors;
    return cloudMesh;
}

// Load default material (Supports: DIFFUSE, SPECULAR, NORMAL maps)
Material LoadMaterialDefault(void)
{
    Material material = { 0 };
    material.maps = (MaterialMap *)RL_CALLOC(MAX_MATERIAL_MAPS, sizeof(MaterialMap));

    material.shader = GetShaderDefault();
    material.maps[MAP_DIFFUSE].texture = GetTextureDefault();   // White texture (1x1 pixel)
    //material.maps[MAP_NORMAL].texture;         // NOTE: By default, not set
    //material.maps[MAP_SPECULAR].texture;       // NOTE: By default, not set

    material.maps[MAP_DIFFUSE].color =  (Color){ 255, 255, 255, 255 };   // Diffuse color
    material.maps[MAP_SPECULAR].color = (Color){ 255, 255, 255, 255 };   // Specular color

    return material;
}

// Load model from generated mesh
// WARNING: A shallow copy of mesh is generated, passed by value,
// as long as struct contains pointers to data and some values, we get a copy
// of mesh pointing to same data as original version... be careful!
Model LoadModelFromMesh(Mesh mesh)
{
    Model model = { 0 };

    model.transform = MatrixIdentity();

    model.meshCount = 1;
    model.meshes = (Mesh *)RL_CALLOC(model.meshCount, sizeof(Mesh));
    model.meshes[0] = mesh;

    model.materialCount = 1;
    model.materials = (Material *)RL_CALLOC(model.materialCount, sizeof(Material));
    model.materials[0] = LoadMaterialDefault();

    model.meshMaterial = (int *)RL_CALLOC(model.meshCount, sizeof(int));
    model.meshMaterial[0] = 0;  // First material index

    return model;
}

}