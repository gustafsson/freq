$framework_path = "/Library/Frameworks"
$cuda_library_path = "/usr/local/cuda/lib"
$custom_library_path = "../../../libs"
$command_line_width = 80

# Configuration
$platform = "macos_i386"
$platform = ARGV[1] if( ARGV[1] and !ARGV[1].match(/^--/))
$version = "dev"
$version = ARGV[0] if( ARGV[0] and !ARGV[0].match(/^--/))
$build_name = "sonicawe_#{$version}_#{$platform}"

$zip = true
$zip = false if(ARGV.index("--nozip"))

def qt_lib_path(name, debug = false)
    return "#{$framework_path}/#{name}.framework/Versions/Current/#{name}#{"_debug" if(debug)}"
end

def qt_install_name(name)
    return "#{name}.framework/Versions/4/#{name}"
end

def cuda_lib_path(name)
    return "#{$cuda_library_path}/lib#{name}.dylib"
end

def custom_lib_path(name, path = nil)
    return "#{$custom_library_path}/#{"#{path}/" if(path)}lib#{name}.dylib"
end

def package_macos(app_name, version, zip = false)
    libraries = [qt_lib_path("QtGui"),
                 qt_lib_path("QtOpenGL"),
                 qt_lib_path("QtCore"),
                 cuda_lib_path("cufft"),
                 cuda_lib_path("cudart"),
                 cuda_lib_path("tlshook"),
                 custom_lib_path("portaudio"),
                 custom_lib_path("portaudiocpp"),
                 custom_lib_path("sndfile"),
                 custom_lib_path("hdf5", "hdf5/bin"),
                 custom_lib_path("hdf5_hl", "hdf5/bin")]
    
    directories = ["Contents/Frameworks",
                   "Contents/MacOS",
                   "Contents/Resources",
                   "Contents/plugins"]
    
    executables = [["../sonicawe", "sonicawe_app"],
                   ["package-macos/launcher", "Sonicawe"]]
    
    resources = ["#{$framework_path}/QtGui.framework/Versions/Current/Resources/qt_menu.nib",
                 "package-macos/aweicon-project.icns",
                 "package-macos/aweicon.icns",
                 "package-macos/qt.conf"]
    
    install_names = [[qt_install_name("QtOpenGL"), "@executable_path/../Frameworks/QtOpenGL"],
                     [qt_install_name("QtGui"), "@executable_path/../Frameworks/QtGui"],
                     [qt_install_name("QtCore"), "@executable_path/../Frameworks/QtCore"],
                     ["@rpath/libtlshook.dylib", "@executable_path/../Frameworks/libtlshook.dylib"],
                     ["@rpath/libcufft.dylib", "@executable_path/../Frameworks/libcufft.dylib"],
                     ["@rpath/libcudart.dylib", "@executable_path/../Frameworks/libcudart.dylib"]]
    
    use_bin = Array.new()
    
    # Creating directories
    puts " Creating application directories ".center($command_line_width, "=")
    directories.each do |directory|
        puts " creating: #{app_name}.app/#{directory}"
        unless system("mkdir -p #{app_name}.app/#{directory}")
            puts "Error: Could not create directory, #{directory}"
            exit(1)
        end
    end
    
    # Copying libraries
    puts " Copying dynamic libraries ".center($command_line_width, "=")
    libraries.each do |library|
        puts " copying: #{library}"
        local_lib = "#{app_name}.app/Contents/Frameworks/#{File.basename(library)}"
        use_bin.push(local_lib)
        unless system("cp #{library} #{local_lib}")
            puts "Error: Could not copy library, #{library}"
            exit(1)
        end
    end
    
    # Copying executables
    puts " Copying executables ".center($command_line_width, "=")
    executables.each do |executable|
        puts " copying: #{executable[0]}"
        local_exec = "#{app_name}.app/Contents/MacOS/#{File.basename(executable[1])}"
        use_bin.push(local_exec)
        unless system("cp #{executable[0]} #{local_exec}")
            puts "Error: Could not copy executable, #{executable[0]}"
            exit(1)
        end
    end
    
    # Copying resources
    puts " Copying resources ".center($command_line_width, "=")
    resources.each do |resource|
        puts " copying: #{resource}"
        unless system("cp -r #{resource} #{app_name}.app/Contents/Resources/#{File.basename(resource)}")
            puts "Error: Could not copy resource, #{resource}"
            exit(1)
        end
    end
    unless system("cp -r ../matlab #{app_name}.app/Contents/MacOS/matlab")
        puts "Error: Could not copy resource, matlab directory"
        exit(1)
    end
    
    # Add application information
    puts " Adding application information ".center($command_line_width, "=")
    puts " writing: Info.plist"
    info = File.read("package-macos/Info.plist")
    info.gsub!("(VERSION_TAG)", "#{version}")
    info.gsub!("(LONG_VERSION_TAG)", "SonicAwe #{version}")
    File.open("#{app_name}.app/Contents/Info.plist", "w") do |file|
        file.write(info)
    end
    puts " copying: package-macos/PkgInfo"
    system("cp package-macos/PkgInfo #{app_name}.app/Contents/PkgInfo")
    
    # Setting install names
    puts " Fixing install names ".center($command_line_width, "=")
    use_bin.each do |path|
        puts " binary: #{path}"
        install_names.each do |install_name|
            puts "  changing: #{install_name[0]}"
            unless system("install_name_tool -change #{install_name[0]} #{install_name[1]} #{path}")
                puts "Error: Could not change install name, #{install_name[0]}, for executable, #{path}"
                exit(1)
            end
        end
    end
    
    # Generating zip file
    if( zip )
        puts " Packaging application ".center($command_line_width, "=")
        if( File.exist?("#{app_name}.zip") )
            puts " removing: #{app_name}.zip"
            system("rm #{app_name}.zip")
        end
        puts " creating: #{app_name}.zip"
        unless system("zip -r #{app_name}.zip  #{app_name}.app") && system("zip -r #{app_name}.zip  ../license")
            puts "Error: Unable to zip application, #{app_name}.app"
            exit(1)
        end
    end
end

package_macos($build_name, $version, $zip)