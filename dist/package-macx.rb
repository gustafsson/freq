$framework_path = "/Library/Frameworks"
$cuda_library_path = "/usr/local/cuda/lib"
$custom_library_path = "../lib/sonicawe-maclib/lib"
$command_line_width = 80

# Configuration
$custom_exec = "../src/sonicawe"
$custom_exec = ARGV[3] if( ARGV[3] and !ARGV[3].match(/^--/) )
$platform = "osx"
$platform = ARGV[2] if( ARGV[2] and !ARGV[2].match(/^--/) )
$version = "dev"
$version = ARGV[1] if( ARGV[1] and !ARGV[1].match(/^--/) )
$build_name = "sonicawe"
$build_name = ARGV[0] if( ARGV[0] and !ARGV[0].match(/^--/) )
$packagename = $build_name + "_" + $version

$zip = true
$zip = false if(ARGV.index("--nozip"))

#def qt_lib_path(name, debug = false)
#    return "#{$framework_path}/#{name}.framework/Versions/Current/#{name}#{"_debug" if(debug)}"
#end

def qt_install_name(name)
    return "#{name}.framework/Versions/4/#{name}"
end

def cuda_lib_path(name)
    return "#{$cuda_library_path}/lib#{name}.dylib"
end

def custom_lib_path(name, path = nil)
    return "#{$custom_library_path}/#{"#{path}/" if(path)}lib#{name}.dylib"
end

def package_macos(app_name, version, packagename, zip = false)
    libraries = [cuda_lib_path("cufft"),
                 cuda_lib_path("cudart"),
                 cuda_lib_path("tlshook"),
                 custom_lib_path("portaudio"),
                 custom_lib_path("portaudiocpp"),
                 custom_lib_path("sndfile"),
                 custom_lib_path("FLAC"),
                 custom_lib_path("ogg"),
                 custom_lib_path("vorbis"),
                 custom_lib_path("vorbisenc"),
                 custom_lib_path("hdf5"),
                 custom_lib_path("hdf5_hl")];

    directories = ["Contents/Frameworks",
                   "Contents/MacOS",
                   "Contents/MacOS/plugins",
                   "Contents/Resources",
                   "Contents/plugins"]

    executables = [[$custom_exec, "sonicawe"],
                   [$custom_exec + "-cuda", "sonicawe-cuda"],
                   ["package-macos~/launcher", "launcher"]]

    resources = ["#{$framework_path}/QtGui.framework/Versions/Current/Resources/qt_menu.nib",
                 "package-macos~/aweicon-project.icns",
                 "package-macos~/aweicon.icns"]

    use_bin = Array.new()

    unless system("rm -rf #{app_name}.app")
        puts "Error: Could not clear #{app_name}.app"
        exit(1)
    end

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
    unless system("cp -r ../plugins #{app_name}.app/Contents/MacOS/plugins/examples")
        puts "Error: Could not copy resource, plugins directory"
        exit(1)
    end

    # Add application information
    puts " Adding application information ".center($command_line_width, "=")
    puts " writing: Info.plist"
    info = File.read("package-macos~/Info.plist")
    info.gsub!("(VERSION_TAG)", "#{version}")
    info.gsub!("(LONG_VERSION_TAG)", "Sonic AWE #{version}")
    File.open("#{app_name}.app/Contents/Info.plist", "w") do |file|
        file.write(info)
    end
    puts " copying: package-macos~/PkgInfo"
    system("cp package-macos~/PkgInfo #{app_name}.app/Contents/PkgInfo")

    # Setting install names
    puts " Fixing install names ".center($command_line_width, "=")
    libraries.each do |library|
        libname = "#{File.basename(library)}"
        puts " library: #{libname}"

        libpath = "#{app_name}.app/Contents/Frameworks"
        newlibpath = "@executable_path/../Frameworks"
        libfile = "#{libpath}/#{libname}"
        targetid = `otool -DX #{libfile}`.strip
        newtargetid = "#{newlibpath}/#{libname}"

        unless system("install_name_tool -id #{newtargetid} #{libfile}")
            puts "Error: Could not set id #{newtargetid} in binary #{libfile}"
            exit(1)
        end

        use_bin.each do |path|
            binary_uses_this_lib = !`otool -L #{path} | grep #{targetid}`.empty?
            next unless binary_uses_this_lib

            puts "  in binary: #{File.basename(path)}"
            unless system("install_name_tool -change #{targetid} #{newtargetid} #{path}")
                puts "Error: Could not change install name for #{libpath}/#{libname} from #{targetid} to #{newtargetid} in binary #{path}"
                exit(1)
            end
        end
    end

    unless system("macdeployqt #{app_name}.app -dmg -executable=#{app_name}.app/Contents/MacOS/#{app_name}-cuda")
        puts "Error: Could not run macdeployqt #{app_name}.app"
        exit(1)
    end

    unless system("mv #{app_name}.dmg #{packagename}.dmg")
        puts "Error: Could not run mv #{app_name}.dmg #{packagename}.dmg"
        exit(1)
    end

    # Generating zip file
    #if( zip )
    #    puts " Packaging application ".center($command_line_width, "=")
    #    if( File.exist?("#{packagename}.zip") )
    #        puts " removing: #{packagename}.zip"
    #        system("rm #{packagename}.zip")
    #    end
    #    puts " creating: #{app_name}.zip"
    #    unless system("zip -r #{packagename}.zip #{app_name}.app") && system("zip -r #{packagename}.zip  ../license")
    #        puts "Error: Unable to zip application, #{app_name}.app"
    #        exit(1)
    #    end
    #end

    # TODO create nice looking dmg
    # http://stackoverflow.com/questions/96882/how-do-i-create-a-nice-looking-dmg-for-mac-os-x-using-command-line-tools
end

package_macos($build_name, $version, $packagename, $zip)
