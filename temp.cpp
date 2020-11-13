DIR *d;
    struct dirent *dir;
    d = opendir("/home/ubuntu/unbeatables/dataset/LARC2020/dataset/");
    if (d)
    {
        while ((dir = readdir(d)) != NULL)
        {
            image_path_bugado.push_back(dir->d_name);
        }
        closedir(d);
    }

    std::cout << image_path_bugado.size()<<std::endl;
    image_path_bugado.erase(image_path_bugado.begin());
    image_path_bugado.erase(image_path_bugado.begin());

    for (int i = 0; i < image_path_bugado.size(); ++i)
    {
        string desbug;
       for (int j = 0;image_path_bugado[i][j] != '\0'; ++j)
       {
           desbug[j] = image_path_bugado[i][j];
       }
        image_path[i] = desbug;
    }